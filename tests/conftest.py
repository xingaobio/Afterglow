"""Shared fixtures: synthetic images and a lightweight fake Context with in-memory state."""

from __future__ import annotations

import dataclasses
import logging
import math
import os
import shutil
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFilter

from afterglow.config import Config, Organize, Paths, Run
from afterglow.stages.base import STAGE_ORDER, FileRecord, StepOutcome

# --------------------------------------------------------------------------- synthetic images
SIZE = (1200, 800)


def make_sharp(seed: int = 1, size: tuple[int, int] = SIZE) -> Image.Image:
    rng = np.random.default_rng(seed)
    w, h = size
    img = Image.new("RGB", size, (120, 130, 140))
    draw = ImageDraw.Draw(img)
    for _ in range(400):
        x, y = int(rng.integers(0, w)), int(rng.integers(0, h))
        dx, dy = int(rng.integers(8, 120)), int(rng.integers(8, 120))
        colour = tuple(int(c) for c in rng.integers(0, 256, 3))
        draw.rectangle([x, y, x + dx, y + dy], fill=colour)
    return img


def make_tilted(angle_deg: float = 3.0, size: tuple[int, int] = SIZE) -> Image.Image:
    """Sky over ground with the horizon sloping down to the right by `angle_deg`, plus parallel fence rails."""
    w, h = size
    slope = math.tan(math.radians(angle_deg))
    img = Image.new("RGB", size, (200, 220, 255))
    draw = ImageDraw.Draw(img)
    y0 = h * 0.5 - slope * w / 2
    draw.polygon([(0, y0), (w, y0 + slope * w), (w, h), (0, h)], fill=(40, 70, 40))
    for k in range(1, 5):
        yy = y0 + k * 60
        draw.line([(0, yy), (w, yy + slope * w)], fill=(250, 250, 250), width=4)
    return img


def exif_with_time(stamp: str, sub: str = "000", make: str = "TestCam") -> Image.Exif:
    ex = Image.Exif()
    ex[0x010F] = make
    ex[0x8769] = {0x9003: stamp, 0x9291: sub}
    return ex


@pytest.fixture
def images(tmp_path: Path) -> SimpleNamespace:
    folder = tmp_path / "images"
    folder.mkdir()
    sharp = make_sharp()
    sharp.save(folder / "sharp.jpg", quality=95)
    sharp.filter(ImageFilter.GaussianBlur(14)).save(folder / "blurred.jpg", quality=95)
    make_tilted(3.0).save(folder / "tilted.jpg", quality=95)
    bursts = []
    for i in range(5):
        total = i * 0.5
        stamp = f"2024:05:12 13:05:{int(total):02d}"
        sub = "500" if total % 1 else "000"
        path = folder / f"burst_{i + 1:02d}.jpg"
        make_sharp(seed=10 + i).save(path, quality=92, exif=exif_with_time(stamp, sub))
        bursts.append(path)
    dark = Image.fromarray(np.full((400, 600, 3), 1, np.uint8))
    bright = Image.fromarray(np.full((400, 600, 3), 255, np.uint8))
    normal = Image.fromarray(np.random.default_rng(3).integers(40, 200, (400, 600, 3), dtype=np.uint8))
    dark.save(folder / "dark.jpg")
    bright.save(folder / "bright.jpg")
    normal.save(folder / "normal.jpg")
    return SimpleNamespace(
        folder=folder, sharp=folder / "sharp.jpg", blurred=folder / "blurred.jpg", tilted=folder / "tilted.jpg",
        bursts=bursts, dark=folder / "dark.jpg", bright=folder / "bright.jpg", normal=folder / "normal.jpg",
    )


# --------------------------------------------------------------------------- fake collaborators
_FILE_FIELDS = {f.name for f in fields(FileRecord)} - {"analysis"}


class FakeState:
    """Dict-backed stand-in for afterglow.state.State covering the methods the stages call."""

    def __init__(self) -> None:
        self.files: dict[int, FileRecord] = {}
        self.steps: dict[tuple[int, str], str] = {}
        self.errors: dict[tuple[int, str], str | None] = {}
        self._analysis: dict[int, dict[str, Any]] = {}
        self._groups: list[dict[str, Any]] = []
        self._tags: dict[int, dict[tuple[str, str], float | None]] = {}
        self._desc: dict[int, dict[str, Any]] = {}
        self._outputs: list[dict[str, Any]] = []
        self.kv: dict[str, str] = {}
        self._next = 1

    # files
    def upsert_file(self, rec: dict[str, Any]) -> int:
        existing = self.file_by_path(rec["path"])
        data = {k: v for k, v in rec.items() if k in _FILE_FIELDS}
        if existing is not None:
            for k, v in data.items():
                setattr(existing, k, v)
            if "missing" not in data:
                existing.missing = False
            return existing.id
        fid = self._next
        self._next += 1
        self.files[fid] = FileRecord(id=fid, **data)
        for stage in STAGE_ORDER:
            if stage != "backup":
                self.steps[(fid, stage)] = "pending"
        return fid

    def _joined(self, rec: FileRecord) -> FileRecord:
        rec.analysis = dict(self._analysis[rec.id]) if rec.id in self._analysis else None
        return rec

    def get_file(self, file_id: int) -> FileRecord | None:
        rec = self.files.get(file_id)
        return self._joined(rec) if rec else None

    def file_by_path(self, path: str) -> FileRecord | None:
        for rec in self.files.values():
            if rec.path == str(path):
                return self._joined(rec)
        return None

    def files_where(self, **filters: Any) -> list[FileRecord]:
        out = []
        for rec in self.files.values():
            ok = True
            for key, value in filters.items():
                if key == "pick":
                    actual: Any = bool((self._analysis.get(rec.id) or {}).get("pick"))
                    value = bool(value)
                else:
                    actual = getattr(rec, key)
                if isinstance(value, (list, tuple, set)):
                    ok &= actual in value
                else:
                    ok &= actual == value
            if ok:
                out.append(self._joined(rec))
        return sorted(out, key=lambda r: r.id)

    def mark_missing(self, present_paths: set[str]) -> int:
        count = 0
        for rec in self.files.values():
            if rec.path not in present_paths and not rec.missing:
                rec.missing = True
                count += 1
        return count

    # steps
    def pending(
        self, stage: str, limit: int, file_ids: list[int] | None = None, requires: tuple[str, ...] = ()
    ) -> list[FileRecord]:
        recs = [
            self._joined(r) for r in self.files.values()
            if self.steps.get((r.id, stage)) == "pending" and not r.missing and (file_ids is None or r.id in file_ids)
        ]
        return sorted(recs, key=lambda r: r.id)[:limit]

    def set_step(self, file_id: int, stage: str, status: str, error: str | None = None) -> None:
        self.steps[(file_id, stage)] = status
        self.errors[(file_id, stage)] = error

    def set_steps(self, outcomes: list[StepOutcome], stage: str) -> None:
        for o in outcomes:
            self.set_step(o.file_id, stage, o.status, o.error)

    def step(self, file_id: int, stage: str) -> str | None:
        return self.steps.get((file_id, stage))

    def reset_stage(self, stage: str, only_failed: bool = True) -> int:
        return 0

    # analysis / groups / tags / descriptions / outputs
    def set_analysis(self, file_id: int, **fields_: Any) -> None:
        self._analysis.setdefault(file_id, {}).update(fields_)

    def analysis(self, file_id: int) -> dict[str, Any] | None:
        return dict(self._analysis[file_id]) if file_id in self._analysis else None

    def create_group(self, kind: str, member_ids: list[int], name: str | None = None, meta: dict | None = None) -> int:
        times = sorted(self.files[m].captured_at for m in member_ids if self.files[m].captured_at)
        gid = len(self._groups) + 1
        self._groups.append({
            "id": gid, "kind": kind, "name": name, "members": list(member_ids), "meta": meta or {},
            "started_at": times[0] if times else None, "ended_at": times[-1] if times else None,
        })
        return gid

    def groups(self, kind: str, unnamed_only: bool = False) -> list[dict[str, Any]]:
        return [dict(g) for g in self._groups if g["kind"] == kind and (not unnamed_only or g["name"] is None)]

    def set_group_name(self, group_id: int, name: str) -> None:
        self._groups[group_id - 1]["name"] = name

    def add_tags(self, file_id: int, tags: list[tuple[str, str, float | None]]) -> None:
        for tag, source, conf in tags:
            self._tags.setdefault(file_id, {})[(tag, source)] = conf

    def tags(self, file_id: int) -> list[tuple[str, str, float | None]]:
        return [(t, s, c) for (t, s), c in sorted(self._tags.get(file_id, {}).items())]

    def set_description(self, file_id: int, **fields_: Any) -> None:
        self._desc.setdefault(file_id, {}).update(fields_)

    def description(self, file_id: int) -> dict[str, Any] | None:
        return dict(self._desc[file_id]) if file_id in self._desc else None

    def add_output(
        self, kind: str, path: str, file_id: int | None = None, group_id: int | None = None, params: dict | None = None
    ) -> int:
        self._outputs.append({
            "id": len(self._outputs) + 1, "file_id": file_id, "group_id": group_id, "kind": kind, "path": str(path),
            "params": params or {},
        })
        return len(self._outputs)

    def outputs(self, file_id: int | None = None, group_id: int | None = None, kind: str | None = None) -> list[dict]:
        return [
            o for o in self._outputs
            if (file_id is None or o["file_id"] == file_id) and (group_id is None or o["group_id"] == group_id)
            and (kind is None or o["kind"] == kind)
        ]

    # kv
    def get(self, key: str, default: str | None = None) -> str | None:
        return self.kv.get(key, default)

    def set(self, key: str, value: str) -> None:
        self.kv[key] = value


class FakeFS:
    """Performs the real filesystem operation and records every call."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, ...]] = []

    def mkdir(self, path: Any) -> None:
        self.calls.append(("mkdir", str(path)))
        Path(path).mkdir(parents=True, exist_ok=True)

    def symlink(self, target: Any, link: Any) -> None:
        self.calls.append(("symlink", str(target), str(link)))
        Path(link).parent.mkdir(parents=True, exist_ok=True)
        os.symlink(target, link)

    def hardlink(self, src: Any, dst: Any) -> None:
        self.calls.append(("hardlink", str(src), str(dst)))
        Path(dst).parent.mkdir(parents=True, exist_ok=True)
        os.link(src, dst)

    def copy(self, src: Any, dst: Any) -> None:
        self.calls.append(("copy", str(src), str(dst)))
        Path(dst).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    def move(self, src: Any, dst: Any, allow_move_from_source: bool = False) -> None:
        self.calls.append(("move", str(src), str(dst)))
        Path(dst).parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))

    def write_text(self, path: Any, text: str, encoding: str = "utf-8") -> None:
        self.calls.append(("write_text", str(path)))
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(text, encoding=encoding)

    def ops(self, name: str) -> list[tuple[str, ...]]:
        return [c for c in self.calls if c[0] == name]


class FakeUndo:
    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []

    def record(self, action: str, src: Any = None, dst: Any = None, meta: dict | None = None) -> int:
        self.records.append({"action": action, "src": src, "dst": dst, "meta": meta or {}})
        return len(self.records)


def fake_ctx(tmp_path: Path, **overrides: Any) -> SimpleNamespace:
    """Context double rooted at tmp_path (source/, work/, output/, library/ are created).

    Keyword overrides replace ctx attributes (state, lumen, llm, immich, fs, undo, dry_run...) or, when the key is a
    Config section name, that config section (e.g. process=Process(...)).
    """
    config_fields = {f.name for f in dataclasses.fields(Config)}
    # lumen / llm / immich exist both as Config sections and as Context clients: a dataclass value is a section
    config_overrides = {
        k: overrides.pop(k) for k in list(overrides)
        if k in config_fields and k != "path" and dataclasses.is_dataclass(overrides[k])
    }
    paths = Paths(
        source=tmp_path / "source", work=tmp_path / "work", output=tmp_path / "work" / "output",
        organized=tmp_path / "library", backup=None,
    )
    config = Config(paths=paths, run=Run(batch_size=50, workers=2), organize=Organize(mode="symlink"))
    config = dataclasses.replace(config, **config_overrides)
    for d in (config.paths.source, config.paths.work, config.paths.output, config.paths.organized):
        d.mkdir(parents=True, exist_ok=True)
    ctx = SimpleNamespace(
        config=config, state=FakeState(), log=logging.getLogger("test"), run_id=1, dry_run=False, lumen=None,
        lumen_lib=None, llm=None, brain_llm=None, immich=None, undo=FakeUndo(), fs=FakeFS(),
    )
    for key, value in overrides.items():
        setattr(ctx, key, value)
    return ctx


def tree_hashes(root: Path) -> dict[str, str]:
    """sha256 of every file under root, for asserting the source tree is untouched."""
    from afterglow.util.hashing import sha256_file

    return {str(p.relative_to(root)): sha256_file(p) for p in sorted(root.rglob("*")) if p.is_file()}


def add_file(ctx: SimpleNamespace, rel: str, **fields: Any) -> FileRecord:
    """Insert a FileRecord (path under config.paths.source) straight into the fake state."""
    path = ctx.config.paths.source / rel
    ext = path.suffix.lower().lstrip(".")
    from afterglow.config import kind_of

    rec = {"path": str(path), "rel_path": rel, "kind": kind_of(ext), "ext": ext, **fields}
    fid = ctx.state.upsert_file(rec)
    return ctx.state.get_file(fid)  # type: ignore[return-value]
