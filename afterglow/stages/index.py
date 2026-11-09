"""Stage 2 - discover files, hash, EXIF, previews, RAW/JPEG pairing, exact duplicates (ARCHITECTURE.md §5 ②)."""

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any

from PIL import Image

from afterglow.config import RAW_EXTENSIONS, kind_of
from afterglow.stages.base import FileRecord, Stage, StepOutcome
from afterglow.util import exif
from afterglow.util.hashing import dhash, sha256_file
from afterglow.util.images import load_preview, save_jpeg

if TYPE_CHECKING:
    from afterglow.context import Context

log = logging.getLogger(__name__)

SKIP_DIRS = {"@eaDir", ".thumbnails", "System Volume Information", "lost+found", "$RECYCLE.BIN"}
PAIRED_SKIP_STAGES = ("cull", "understand", "process")
PREVIEW_QUALITY = 90
PREVIEW_EDGE = 2560
JPEG_LIKE = ("jpeg", "heic")


def _same_mtime(a: float | None, b: float) -> bool:
    return a is not None and abs(a - b) < 1e-3


class Index(Stage):
    name = "index"
    requires = ()
    uses_gpu = False
    per_file = True

    def __init__(self) -> None:
        self._discovered_run: int | None = None

    # ------------------------------------------------------------------ discovery
    def select(self, ctx: Context, limit: int) -> list[FileRecord]:
        self._ensure_discovered(ctx)
        return ctx.state.pending(self.name, limit)

    def _ensure_discovered(self, ctx: Context) -> None:
        if self._discovered_run == ctx.run_id and self._discovered_run is not None:
            return
        self.discover(ctx)
        self._discovered_run = ctx.run_id

    def discover(self, ctx: Context) -> int:
        """Walk the source tree, upsert new/changed files, mark vanished ones missing; returns files seen."""
        source = ctx.config.paths.source
        allowed = {e.lower().lstrip(".") for e in ctx.config.paths.include_ext}
        present: set[str] = set()
        new = changed = 0
        for dirpath, dirnames, filenames in os.walk(source, followlinks=False):
            dirnames[:] = sorted(d for d in dirnames if not d.startswith(".") and d not in SKIP_DIRS)
            for name in sorted(filenames):
                if name.startswith("."):
                    continue
                ext = Path(name).suffix.lower().lstrip(".")
                if ext not in allowed:
                    continue
                full = Path(dirpath) / name
                try:
                    st = full.stat()
                except OSError:
                    log.warning("无法读取文件信息: %s", full)
                    continue
                path = str(full)
                present.add(path)
                existing = ctx.state.file_by_path(path)
                if existing is not None and existing.size == st.st_size and _same_mtime(existing.mtime, st.st_mtime):
                    if existing.missing:
                        ctx.state.upsert_file({"path": path, "missing": 0})
                    continue
                rec: dict[str, Any] = {
                    "path": path, "rel_path": str(full.relative_to(source)), "size": st.st_size, "mtime": st.st_mtime,
                    "kind": kind_of(ext), "ext": ext,
                }
                if existing is None:
                    new += 1
                else:
                    changed += 1
                    rec.update(sha256=None, dhash=None, preview_path=None, pair_id=None)
                file_id = ctx.state.upsert_file(rec)
                if existing is not None:
                    ctx.state.set_step(file_id, self.name, "pending")
        gone = ctx.state.mark_missing(present)
        log.info("扫描完成: 共 %d 个文件，新增 %d，变化 %d，消失 %d", len(present), new, changed, gone)
        return len(present)

    # ------------------------------------------------------------------ processing
    def run(self, ctx: Context, batch: list[FileRecord]) -> list[StepOutcome]:
        self._ensure_discovered(ctx)
        if not batch:
            return []
        workers = max(1, ctx.config.run.workers)
        outcomes: dict[int, StepOutcome] = {}
        extracted: dict[int, dict[str, Any]] = {}
        with ThreadPoolExecutor(workers) as pool:
            futures = {rec.id: pool.submit(self._hash_and_preview, ctx, rec) for rec in batch}
            exif_map = exif.read_batch([Path(r.path) for r in batch])  # exiftool overlaps with the pool
            for rec in batch:
                try:
                    extracted[rec.id] = futures[rec.id].result()
                except Exception as exc:
                    log.exception("索引失败: %s", rec.path)
                    outcomes[rec.id] = StepOutcome.failed(rec.id, exc)

        stored: list[FileRecord] = []
        for rec in batch:
            if rec.id in outcomes:
                continue
            try:
                fields = self._merge_fields(rec, extracted[rec.id], exif_map.get(rec.path, {}))
                ctx.state.upsert_file({"path": rec.path, "rel_path": rec.rel_path, "kind": rec.kind, "ext": rec.ext,
                                       **fields})
                refreshed = ctx.state.get_file(rec.id)
                if refreshed is not None:
                    stored.append(refreshed)
            except Exception as exc:
                log.exception("保存索引结果失败: %s", rec.path)
                outcomes[rec.id] = StepOutcome.failed(rec.id, exc)

        for rec in stored:
            try:
                self._pair(ctx, rec)
                self._mark_duplicate(ctx, rec)
                outcomes[rec.id] = StepOutcome.done(rec.id)
            except Exception as exc:
                log.exception("配对/去重失败: %s", rec.path)
                outcomes[rec.id] = StepOutcome.failed(rec.id, exc)
        return [outcomes[r.id] for r in batch]

    def _hash_and_preview(self, ctx: Context, rec: FileRecord) -> dict[str, Any]:
        """sha256, preview file and dhash for one file (thread-pool worker); skips work already stored."""
        st = os.stat(rec.path)
        unchanged = rec.size == st.st_size and _same_mtime(rec.mtime, st.st_mtime)
        out: dict[str, Any] = {"size": st.st_size, "mtime": st.st_mtime}
        sha = rec.sha256 if unchanged and rec.sha256 else sha256_file(rec.path)
        out["sha256"] = sha
        preview = Path(rec.preview_path) if rec.preview_path else None
        if unchanged and rec.sha256 and preview is not None and preview.exists() and (rec.dhash or rec.kind == "video"):
            out.update(preview_path=str(preview), dhash=rec.dhash)
            return out
        target = ctx.config.paths.previews / sha[:2] / f"{sha}.jpg"
        img: Image.Image | None = None
        if rec.kind != "other":
            if target.exists():
                with Image.open(target) as existing:
                    existing.load()
                    img = existing.convert("RGB")
            else:
                img = load_preview(rec.path, rec.kind, PREVIEW_EDGE)
                if img is not None:
                    orig = img.info.get("orig_size")
                    save_jpeg(img, target, PREVIEW_QUALITY)
                    if orig:
                        out["orig_size"] = orig
        if img is not None:
            out["preview_path"] = str(target)
            if rec.kind != "video":
                out["dhash"] = dhash(img)
        return out

    @staticmethod
    def _merge_fields(rec: FileRecord, extracted: dict[str, Any], tags: dict[str, Any]) -> dict[str, Any]:
        fields: dict[str, Any] = {k: extracted[k] for k in ("size", "mtime", "sha256", "preview_path", "dhash")
                                  if extracted.get(k) is not None}
        fields.update(tags)
        if not fields.get("captured_at"):
            from_name = exif.captured_from_filename(Path(rec.path).name)
            if from_name:
                fields["captured_at"] = from_name
                fields["captured_src"] = "filename"
            else:
                fields["captured_at"] = exif.mtime_iso(extracted.get("mtime") or rec.mtime or 0.0)
                fields["captured_src"] = "mtime"
        if not (fields.get("width") and fields.get("height")) and extracted.get("orig_size"):
            fields["width"], fields["height"] = extracted["orig_size"]
        return fields

    # ------------------------------------------------------------------ pairing / duplicates
    def _pair(self, ctx: Context, rec: FileRecord) -> None:
        """RAW + JPEG/HEIC with the same stem, same folder and capture time within 1 s -> JPEG.pair_id = RAW.id."""
        if rec.kind != "raw" and rec.kind not in JPEG_LIKE:
            return
        folder = Path(rec.path).parent
        stem = Path(rec.path).stem.lower()
        try:
            names = os.listdir(folder)
        except OSError:
            return
        want_raw = rec.kind != "raw"
        for name in names:
            p = Path(name)
            if p.stem.lower() != stem or name == Path(rec.path).name:
                continue
            ext = p.suffix.lower().lstrip(".")
            if want_raw and ext not in RAW_EXTENSIONS:
                continue
            if not want_raw and kind_of(ext) not in JPEG_LIKE:
                continue
            other = ctx.state.file_by_path(str(folder / name))
            if other is None or other.sha256 is None:
                continue
            raw, jpeg = (other, rec) if want_raw else (rec, other)
            if not self._within_one_second(raw.captured_at, jpeg.captured_at):
                continue
            if jpeg.pair_id == raw.id:
                continue
            ctx.state.upsert_file({"path": jpeg.path, "pair_id": raw.id})
            for stage in PAIRED_SKIP_STAGES:
                ctx.state.set_step(jpeg.id, stage, "skipped")

    @staticmethod
    def _within_one_second(a: str | None, b: str | None) -> bool:
        da, db = exif.parse_iso(a), exif.parse_iso(b)
        if da is None or db is None:
            return True  # same folder and stem is good enough when a time is unknown
        return abs((da - db).total_seconds()) <= 1.0

    def _mark_duplicate(self, ctx: Context, rec: FileRecord) -> None:
        """Later copies of a byte-identical file get reject_reason=duplicate and join a `duplicate` group."""
        if not rec.sha256:
            return
        same = [f for f in ctx.state.files_where(sha256=rec.sha256, missing=False)]
        if len(same) < 2:
            return
        original = min(same, key=lambda f: f.id)
        if original.id == rec.id:
            return
        existing = ctx.state.analysis(rec.id) or {}
        if existing.get("reject_reason") == "duplicate":
            return
        ctx.state.set_analysis(rec.id, reject_reason="duplicate", pick=0)
        ctx.state.create_group("duplicate", [original.id, rec.id], meta={"sha256": rec.sha256, "original": original.id})
        log.info("重复文件: %s 与 %s 相同", rec.rel_path, original.rel_path)


STAGE = Index
