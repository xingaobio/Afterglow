import os
import time

import pytest

from afterglow.stages import index as index_mod
from afterglow.stages.index import STAGE, Index
from afterglow.util import images as images_util
from tests.conftest import fake_ctx, make_sharp, tree_hashes


@pytest.fixture
def library(tmp_path, monkeypatch):
    ctx = fake_ctx(tmp_path)
    src = ctx.config.paths.source
    for d in ("a", "b", ".hidden", "@eaDir"):
        (src / d).mkdir()
    make_sharp(1).save(src / "a" / "IMG_20240512_130501.jpg")
    (src / "a" / "dup.jpg").write_bytes((src / "a" / "IMG_20240512_130501.jpg").read_bytes())
    raw = src / "b" / "IMG_0001.DNG"
    make_sharp(2).save(raw, format="JPEG")  # JPEG bytes inside a .dng: the RAW loader is mocked below
    make_sharp(3).save(src / "b" / "IMG_0001.jpg")
    (src / "b" / "clip.mp4").write_bytes(b"not really a video")
    make_sharp(4).save(src / ".hidden" / "skip.jpg")
    make_sharp(5).save(src / "@eaDir" / "skip.jpg")
    (src / "notes.txt").write_text("ignore me")

    real_load = images_util.load_preview

    def fake_load(path, kind, long_edge=2560):
        if kind == "video":
            return None
        return real_load(path, "jpeg" if kind == "raw" else kind, long_edge)

    monkeypatch.setattr(index_mod, "load_preview", fake_load)

    def fake_exif(paths):
        out = {}
        for p in paths:
            if p.name.startswith("IMG_0001"):
                out[str(p)] = {"captured_at": "2024-05-13T10:00:00.400Z", "captured_src": "exif", "camera": "TestCam",
                               "iso": 100, "width": 1200, "height": 800}
        return out

    monkeypatch.setattr(index_mod.exif, "read_batch", fake_exif)
    return ctx


def run_index(ctx, stage=None):
    stage = stage or Index()
    batch = stage.select(ctx, 100)
    outcomes = stage.run(ctx, batch) if batch else []
    ctx.state.set_steps(outcomes, "index")
    return batch, outcomes


def test_stage_registry():
    assert STAGE is Index and Index.name == "index" and Index.requires == ()


def test_index_discovers_hashes_and_fills_columns(library):
    ctx = library
    before = tree_hashes(ctx.config.paths.source)
    batch, outcomes = run_index(ctx)
    assert len(batch) == 5 and all(o.status == "done" for o in outcomes)
    assert len(outcomes) == len(batch)

    recs = {r.rel_path: r for r in ctx.state.files_where()}
    assert set(recs) == {
        "a/IMG_20240512_130501.jpg", "a/dup.jpg", "b/IMG_0001.DNG", "b/IMG_0001.jpg", "b/clip.mp4",
    }
    jpg = recs["a/IMG_20240512_130501.jpg"]
    assert jpg.kind == "jpeg" and jpg.sha256 and len(jpg.dhash) == 16
    assert jpg.captured_at == "2024-05-12T13:05:01.000Z" and jpg.captured_src == "filename"
    assert (jpg.width, jpg.height) == (1200, 800)
    assert os.path.exists(jpg.preview_path) and jpg.sha256[:2] in jpg.preview_path
    raw = recs["b/IMG_0001.DNG"]
    assert raw.kind == "raw" and raw.captured_src == "exif" and raw.camera == "TestCam"
    video = recs["b/clip.mp4"]
    assert video.kind == "video" and video.preview_path is None and video.dhash is None
    assert tree_hashes(ctx.config.paths.source) == before  # originals untouched


def test_index_pairs_raw_and_jpeg(library):
    ctx = library
    run_index(ctx)
    recs = {r.rel_path: r for r in ctx.state.files_where()}
    raw, jpeg = recs["b/IMG_0001.DNG"], recs["b/IMG_0001.jpg"]
    assert jpeg.pair_id == raw.id and raw.pair_id is None
    for stage in ("cull", "understand", "process"):
        assert ctx.state.step(jpeg.id, stage) == "skipped"
    assert ctx.state.step(raw.id, "cull") == "pending"


def test_index_marks_exact_duplicates(library):
    ctx = library
    run_index(ctx)
    recs = {r.rel_path: r for r in ctx.state.files_where()}
    original, dup = recs["a/IMG_20240512_130501.jpg"], recs["a/dup.jpg"]
    assert ctx.state.analysis(dup.id)["reject_reason"] == "duplicate"
    assert ctx.state.analysis(original.id) is None
    groups = ctx.state.groups("duplicate")
    assert len(groups) == 1 and groups[0]["members"] == [original.id, dup.id]


def test_index_is_rerunnable_and_skips_unchanged(library, monkeypatch):
    ctx = library
    run_index(ctx)
    calls = []
    real = index_mod.sha256_file
    monkeypatch.setattr(index_mod, "sha256_file", lambda p, *a, **k: calls.append(p) or real(p, *a, **k))

    ctx.run_id = 2
    batch, _ = run_index(ctx, Index())
    assert batch == [] and calls == []

    # a changed file is re-indexed, the rest is left alone
    target = ctx.config.paths.source / "a" / "dup.jpg"
    time.sleep(0.01)
    make_sharp(77).save(target)
    ctx.run_id = 3
    batch, outcomes = run_index(ctx, Index())
    assert [r.rel_path for r in batch] == ["a/dup.jpg"]
    assert [str(target)] == [str(p) for p in calls]
    rec = ctx.state.file_by_path(str(target))
    assert rec.sha256 != ctx.state.file_by_path(str(ctx.config.paths.source / "a" / "IMG_20240512_130501.jpg")).sha256


def test_index_marks_vanished_files_missing(library):
    ctx = library
    run_index(ctx)
    (ctx.config.paths.source / "a" / "dup.jpg").unlink()
    ctx.run_id = 2
    run_index(ctx, Index())
    assert ctx.state.file_by_path(str(ctx.config.paths.source / "a" / "dup.jpg")).missing is True


def test_index_reports_failure_per_file(library, monkeypatch):
    ctx = library
    real = index_mod.sha256_file

    def flaky(path, *a, **k):
        if str(path).endswith("clip.mp4"):
            raise OSError("disk error")
        return real(path, *a, **k)

    monkeypatch.setattr(index_mod, "sha256_file", flaky)
    batch, outcomes = run_index(ctx)
    by = {o.file_id: o for o in outcomes}
    failed = [o for o in outcomes if o.status == "failed"]
    assert len(failed) == 1 and "disk error" in failed[0].error
    assert sum(1 for o in by.values() if o.status == "done") == 4
