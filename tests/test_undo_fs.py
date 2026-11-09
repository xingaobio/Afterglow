from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import pytest

from afterglow import config
from afterglow.context import Context
from afterglow.state import State
from afterglow.util.fs import SourceProtectedError, atomic_write_bytes, is_under, unique_path


def make_ctx(tmp_path: Path, dry_run: bool = False) -> Context:
    src = tmp_path / "source"
    src.mkdir(exist_ok=True)
    cfg = config.from_dict({
        "paths": {
            "source": str(src), "work": str(tmp_path / "work"), "output": str(tmp_path / "out"),
            "organized": str(tmp_path / "lib"),
        },
        "gpu": {"lock_file": str(tmp_path / "gpu.lock")},
    })
    state = State(tmp_path / "work" / "afterglow.db")
    ctx = Context(config=cfg, state=state, log=logging.getLogger("test"), dry_run=dry_run)
    ctx.run_id = state.start_run(["publish"])
    return ctx


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def test_roundtrip_all_actions(tmp_path: Path) -> None:
    ctx = make_ctx(tmp_path)
    original = tmp_path / "source" / "a.cr3"
    original.write_bytes(b"RAWDATA")
    before = sha(original)
    lib = tmp_path / "lib"

    ctx.fs.symlink(original, lib / "2024" / "ev" / "a.cr3")
    ctx.fs.hardlink(original, lib / "2024" / "ev" / "a_hard.cr3")
    ctx.fs.copy(original, lib / "copy" / "a.cr3")
    ctx.fs.write_text(lib / "note.txt", "hi")
    scratch = tmp_path / "work" / "scratch.bin"
    scratch.parent.mkdir(parents=True, exist_ok=True)
    scratch.write_bytes(b"x")
    ctx.fs.move(scratch, lib / "moved.bin")

    assert (lib / "2024" / "ev" / "a.cr3").is_symlink()
    assert (lib / "moved.bin").exists() and not scratch.exists()

    report = ctx.undo.undo(last=True)
    assert report.errors == []
    assert report.undone >= 9
    assert not (lib / "2024").exists() and not (lib / "copy").exists()
    assert not (lib / "note.txt").exists()
    assert scratch.read_bytes() == b"x"
    assert sha(original) == before
    assert ctx.state.undo_entries() == []


def test_undo_dry_run_changes_nothing(tmp_path: Path) -> None:
    ctx = make_ctx(tmp_path)
    ctx.fs.write_text(tmp_path / "lib" / "f.txt", "x")
    report = ctx.undo.undo(run_id=ctx.run_id, dry_run=True)
    assert report.planned and report.undone == 0
    assert (tmp_path / "lib" / "f.txt").exists()
    assert ctx.state.undo_entries()


def test_undo_mkdir_keeps_nonempty_and_collects_errors(tmp_path: Path) -> None:
    ctx = make_ctx(tmp_path)
    ctx.fs.mkdir(tmp_path / "lib" / "d")
    (tmp_path / "lib" / "d" / "foreign.txt").write_text("keep")
    # a move whose origin got occupied -> error, but other entries still processed
    a = tmp_path / "work" / "m.bin"
    a.parent.mkdir(parents=True, exist_ok=True)
    a.write_bytes(b"1")
    ctx.fs.move(a, tmp_path / "lib" / "d" / "m.bin")
    a.write_bytes(b"2")
    report = ctx.undo.undo(run_id=ctx.run_id)
    assert len(report.errors) == 1
    assert (tmp_path / "lib" / "d" / "foreign.txt").exists()
    assert (tmp_path / "lib" / "d").exists()


def test_undo_immich_entries(tmp_path: Path) -> None:
    ctx = make_ctx(tmp_path)
    calls: list[tuple] = []

    class FakeImmich:
        def delete_album(self, album_id: str) -> None:
            calls.append(("delete_album", album_id))

        def untag_assets(self, tag_id: str, ids: list[str]) -> None:
            calls.append(("untag", tag_id, ids))

        def update_asset(self, asset_id: str, description: str | None = None, rating: int | None = None) -> dict:
            calls.append(("update", asset_id, description))
            return {}

    ctx._cache["immich"] = FakeImmich()
    ctx.undo.record("immich_album", meta={"album_id": "al1"})
    ctx.undo.record("immich_add_assets", meta={"album_id": "al1", "asset_ids": ["x"]})
    ctx.undo.record("immich_tag", meta={"tag_id": "t1", "asset_ids": ["x"]})
    ctx.undo.record("immich_description", meta={"asset_id": "x", "previous": "old"})
    report = ctx.undo.undo(run_id=ctx.run_id)
    assert report.errors == []
    assert calls == [("update", "x", "old"), ("untag", "t1", ["x"]), ("delete_album", "al1")]


def test_source_protection(tmp_path: Path) -> None:
    ctx = make_ctx(tmp_path)
    original = tmp_path / "source" / "a.jpg"
    original.write_bytes(b"J")
    with pytest.raises(SourceProtectedError):
        ctx.fs.write_text(tmp_path / "source" / "new.txt", "x")
    with pytest.raises(SourceProtectedError):
        ctx.fs.symlink(tmp_path / "lib", tmp_path / "source" / "link")
    with pytest.raises(SourceProtectedError):
        ctx.fs.copy(original, tmp_path / "source" / "copy.jpg")
    with pytest.raises(SourceProtectedError):
        ctx.fs.mkdir(tmp_path / "source" / "sub")
    with pytest.raises(SourceProtectedError):
        ctx.fs.move(original, tmp_path / "lib" / "a.jpg")
    assert original.exists()
    ctx.fs.move(original, tmp_path / "lib" / "a.jpg", allow_move_from_source=True)
    assert (tmp_path / "lib" / "a.jpg").exists()
    ctx.undo.undo(last=True)
    assert original.read_bytes() == b"J"


def test_dry_run_and_existing_target(tmp_path: Path) -> None:
    ctx = make_ctx(tmp_path, dry_run=True)
    ctx.fs.write_text(tmp_path / "lib" / "x.txt", "x")
    ctx.fs.mkdir(tmp_path / "lib" / "y")
    assert not (tmp_path / "lib").exists()
    assert ctx.state.undo_entries() == []
    ctx.dry_run = False
    ctx.fs.write_text(tmp_path / "lib" / "x.txt", "x")
    with pytest.raises(FileExistsError):
        ctx.fs.copy(tmp_path / "lib" / "x.txt", tmp_path / "lib" / "x.txt")


def test_helpers(tmp_path: Path) -> None:
    assert is_under(tmp_path / "a" / "b", tmp_path)
    assert is_under(tmp_path, tmp_path)
    assert not is_under(tmp_path.parent, tmp_path)
    assert not is_under(tmp_path / "ab", tmp_path / "a")
    # a symlink inside root pointing outside is still "under" root (not followed)
    link = tmp_path / "ln"
    link.symlink_to(tmp_path.parent)
    assert is_under(link, tmp_path)

    p = tmp_path / "f.bin"
    atomic_write_bytes(p, b"1")
    atomic_write_bytes(p, b"22")
    assert p.read_bytes() == b"22"
    assert [x.name for x in tmp_path.iterdir() if x.name.startswith(".f.bin")] == []

    assert unique_path(tmp_path / "free.jpg", "abcdef123") == tmp_path / "free.jpg"
    q = unique_path(p, "abcdef123")
    assert q.name == "f_abcdef.bin"
    q.write_bytes(b"")
    assert unique_path(p, "abcdef123").name == "f_abcdef_2.bin"
