"""Stage 9 - XMP sidecars, organized tree and Immich sync (ARCHITECTURE.md §5 ⑨)."""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any
from xml.sax.saxutils import escape, quoteattr

from afterglow.stages.base import FileRecord, Stage, StepOutcome
from afterglow.stages.understand import sanitize_name
from afterglow.util import exif

if TYPE_CHECKING:
    from afterglow.context import Context

log = logging.getLogger(__name__)

SCATTERED = "零散"
IMMICH_CONTAINER_ROOT = "/mnt/afterglow"
IMMICH_WAIT_SECONDS = 1800
BACKUP_FRESH = timedelta(hours=24)

XMP_TEMPLATE = """<?xpacket begin="﻿" id="W5M0MpCehiHzreSzNTczkc9d"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/" x:xmptk="Afterglow">
 <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about=""
    xmlns:xmp="http://ns.adobe.com/xap/1.0/"
    xmlns:dc="http://purl.org/dc/elements/1.1/"
    xmp:Rating="{rating}"{label}>{description}{subject}
  </rdf:Description>
 </rdf:RDF>
</x:xmpmeta>
<?xpacket end="w"?>
"""


def rating_for(analysis: dict[str, Any]) -> int:
    """Star rating: quality >= 85 -> 5, burst_best -> 4, pick -> 3, otherwise 0; rejected frames get 0."""
    if analysis.get("reject_reason"):
        return 0
    if (analysis.get("quality") or 0) >= 85:
        return 5
    if analysis.get("burst_best"):
        return 4
    if analysis.get("pick"):
        return 3
    return 0


def build_xmp(analysis: dict[str, Any], tags: list[str], caption: str | None) -> str:
    """XMP packet with xmp:Rating / xmp:Label, dc:description and dc:subject."""
    label = f" xmp:Label={quoteattr('Rejected')}" if analysis.get("reject_reason") else ""
    description = ""
    if caption:
        description = (
            f'\n   <dc:description><rdf:Alt><rdf:li xml:lang="x-default">{escape(caption)}</rdf:li></rdf:Alt>'
            "</dc:description>"
        )
    subject = ""
    if tags:
        items = "".join(f"<rdf:li>{escape(t)}</rdf:li>" for t in tags)
        subject = f"\n   <dc:subject><rdf:Bag>{items}</rdf:Bag></dc:subject>"
    return XMP_TEMPLATE.format(rating=rating_for(analysis), label=label, description=description, subject=subject)


@dataclass
class _Placement:
    rec: FileRecord
    main: FileRecord
    bucket: str  # keep | culled | duplicate
    dst: Path | None = None


class Publish(Stage):
    name = "publish"
    requires = ("understand",)
    uses_gpu = False
    per_file = False

    def run(self, ctx: Context, batch: list[FileRecord]) -> list[StepOutcome]:
        state = ctx.state
        files = state.files_where(missing=False)
        by_id = {f.id: f for f in files}
        events = self._event_index(ctx)
        analyses = {f.id: (f.analysis or state.analysis(f.id) or {}) for f in files}

        failed: dict[int, StepOutcome] = {}
        for rec in files:
            if rec.pair_id is None and rec.is_image:
                try:
                    self._write_xmp(ctx, rec, analyses[rec.id])
                except Exception as exc:
                    log.exception("XMP 写入失败: %s", rec.path)
                    failed[rec.id] = StepOutcome.failed(rec.id, exc)

        placements = self._build_tree(ctx, files, by_id, analyses, events, failed)
        if ctx.immich is not None and not ctx.dry_run:
            try:
                self._immich(ctx, placements, analyses, events)
            except Exception:
                log.exception("Immich 同步失败，目录树不受影响")
        elif ctx.immich is not None:
            log.info("[dry-run] 跳过 Immich 同步")

        outcomes = [failed.get(f.id) or StepOutcome.done(f.id) for f in files]
        state.set_steps(outcomes, self.name)
        log.info("发布完成: %d 个文件，%d 个失败", len(files), len(failed))
        return []

    # ------------------------------------------------------------------ events
    @staticmethod
    def _event_index(ctx: Context) -> dict[int, dict[str, Any]]:
        index: dict[int, dict[str, Any]] = {}
        for g in ctx.state.groups("event"):
            for fid in g["members"]:
                index[fid] = g
        return index

    # ------------------------------------------------------------------ XMP
    def _write_xmp(self, ctx: Context, rec: FileRecord, analysis: dict[str, Any]) -> None:
        state = ctx.state
        tags = [t[0] for t in state.tags(rec.id)]
        description = state.description(rec.id) or {}
        if not analysis and not tags and not description:
            return
        text = build_xmp(analysis, tags, description.get("caption"))
        rel = Path(rec.rel_path)
        out = ctx.config.paths.output / "xmp" / rel.parent / f"{rel.stem}.xmp"
        if out.exists() and out.read_text(encoding="utf-8") != text and any(
            o["path"] == str(out) and o["file_id"] != rec.id for o in state.outputs(kind="xmp")
        ):
            out = out.with_name(f"{rel.stem}_{rec.ext}.xmp")  # same stem, different file in the folder
        if out.exists() and out.read_text(encoding="utf-8") == text:
            return
        ctx.fs.write_text(out, text)
        if not ctx.dry_run:
            state.add_output("xmp", str(out), file_id=rec.id, params={"rating": rating_for(analysis)})

    # ------------------------------------------------------------------ organized tree
    def _effective_mode(self, ctx: Context) -> str:
        mode = ctx.config.organize.mode
        if mode != "move" or not ctx.config.backup.require_before_move:
            return mode
        stamp = exif.parse_iso(ctx.state.get("last_backup_ok"))
        if stamp is not None and datetime.now(UTC) - stamp <= BACKUP_FRESH:
            return mode
        log.warning("24 小时内没有通过校验的备份，move 降级为 symlink")
        return "symlink"

    def _bucket(self, main: FileRecord, analysis: dict[str, Any]) -> str:
        if analysis.get("reject_reason") == "duplicate":
            return "duplicate"
        if main.is_video or not analysis or analysis.get("pick"):
            return "keep"
        return "culled"

    def _build_tree(
        self,
        ctx: Context,
        files: list[FileRecord],
        by_id: dict[int, FileRecord],
        analyses: dict[int, dict[str, Any]],
        events: dict[int, dict[str, Any]],
        failed: dict[int, StepOutcome],
    ) -> list[_Placement]:
        mode = self._effective_mode(ctx)
        used: dict[Path, str] = {}
        placements: list[_Placement] = []
        original_of: dict[int, int] = {}  # duplicate -> the copy it duplicates (shares its event and date)
        for g in ctx.state.groups("duplicate"):
            original = (g.get("meta") or {}).get("original")
            for fid in g["members"]:
                if original is not None and fid != original:
                    original_of[fid] = int(original)
        for rec in files:
            main = by_id.get(rec.pair_id) if rec.pair_id is not None else rec
            if main is None:
                main = rec
            anchor = by_id.get(original_of.get(rec.id, -1), main)
            bucket = self._bucket(main, analyses.get(main.id, {}))
            place = _Placement(rec, main, bucket)
            try:
                dst = self._target(ctx, rec, anchor, bucket, events, used)
                # copy_picks: keepers get a real copy on the SSD (viewable with the drive unplugged),
                # everything else stays a cheap symlink.
                file_mode = mode
                if mode == "copy_picks":
                    file_mode = "copy" if bucket == "keep" else "symlink"
                self._link(ctx, Path(rec.path), dst, file_mode)
                place.dst = dst
                if mode == "move" and not ctx.dry_run and dst.exists() and not Path(rec.path).exists():
                    # The original now lives in the organized tree; keep the catalog pointing at it.
                    relocate = getattr(ctx.state, "relocate_file", None)
                    if callable(relocate):
                        relocate(rec.id, str(dst), str(dst.relative_to(ctx.config.paths.organized)))
                if rec.id == main.id and bucket == "keep":
                    self._link_exports(ctx, rec, dst, used, mode)
            except Exception as exc:
                log.exception("整理失败: %s", rec.path)
                failed[rec.id] = StepOutcome.failed(rec.id, exc)
            placements.append(place)
        return placements

    def _target(
        self, ctx: Context, rec: FileRecord, main: FileRecord, bucket: str, events: dict[int, dict[str, Any]],
        used: dict[Path, str],
    ) -> Path:
        org = ctx.config.organize
        group = events.get(main.id)
        event = sanitize_name((group or {}).get("name") or SCATTERED, 60) or SCATTERED
        when = exif.parse_iso(main.captured_at or rec.captured_at) or datetime(1970, 1, 1, tzinfo=UTC)
        name = Path(rec.path).name
        relative = org.layout.format(
            year=when.year, month=when.month, day=when.day, event=event, basename=name,
            stem=Path(name).stem, ext=Path(name).suffix.lstrip("."),
        )
        base = ctx.config.paths.organized
        if bucket == "culled":
            base = base / org.culled_dir
        elif bucket == "duplicate":
            base = base / org.duplicates_dir
        return self._unique(base / relative, rec, used)

    @staticmethod
    def _unique(dst: Path, rec: FileRecord, used: dict[Path, str]) -> Path:
        """Same-name conflicts between different source files get `_<sha[:6]>` before the extension."""
        candidate = dst
        if candidate in used and used[candidate] != rec.path:
            tag = (rec.sha256 or hashlib.sha256(rec.path.encode()).hexdigest())[:6]
            candidate = dst.with_name(f"{dst.stem}_{tag}{dst.suffix}")
        used[candidate] = rec.path
        return candidate

    @staticmethod
    def _already_linked(src: Path, dst: Path) -> bool:
        if dst.is_symlink():
            return Path(dst.readlink()) == src
        if not dst.exists():
            return False
        try:
            return dst.samefile(src) or dst.stat().st_size == src.stat().st_size
        except OSError:
            return False

    def _link(self, ctx: Context, src: Path, dst: Path, mode: str) -> None:
        if self._already_linked(src, dst):
            return
        if dst.is_symlink() or dst.exists():
            raise FileExistsError(f"目标已存在且内容不同: {dst}")
        if mode == "move":
            ctx.fs.move(src, dst, allow_move_from_source=True)
        else:
            getattr(ctx.fs, mode)(src, dst)

    def _link_exports(self, ctx: Context, rec: FileRecord, dst: Path, used: dict[Path, str], mode: str) -> None:
        """The process export (and nothing else) sits next to its RAW as <stem>_edit.jpg."""
        exports = [o for o in ctx.state.outputs(file_id=rec.id, kind="export") if Path(o["path"]).exists()]
        if not exports:
            return
        src = Path(exports[-1]["path"])
        target = dst.with_name(f"{dst.stem}_edit{src.suffix}")
        if target in used and used[target] != str(src):
            target = target.with_name(f"{target.stem}_{(rec.sha256 or '')[:6]}{target.suffix}")
        used[target] = str(src)
        self._link(ctx, src, target, {"move": "symlink", "copy_picks": "copy"}.get(mode, mode))

    # ------------------------------------------------------------------ Immich
    @staticmethod
    def _find_asset(immich: Any, dst: Path, organized: Path, cache: dict[str, dict | None]) -> dict | None:
        key = str(dst)
        if key not in cache:
            asset = immich.asset_by_path(key)
            if asset is None:
                try:
                    inside = f"{IMMICH_CONTAINER_ROOT}/{dst.relative_to(organized).as_posix()}"
                except ValueError:
                    inside = None
                asset = immich.asset_by_path(inside) if inside else None
            cache[key] = asset
        return cache[key]

    def _immich(
        self, ctx: Context, placements: list[_Placement], analyses: dict[int, dict[str, Any]],
        events: dict[int, dict[str, Any]],
    ) -> None:
        immich, state, cfg = ctx.immich, ctx.state, ctx.config
        assert immich is not None
        library_id = state.get("immich_library_id")
        if not library_id:
            for lib in immich.libraries():
                if lib.get("name") == cfg.immich.external_library_name:
                    library_id = str(lib["id"])
                    break
            else:
                library_id = immich.create_external_library(
                    cfg.immich.external_library_name, [str(cfg.paths.organized)]
                )
            state.set("immich_library_id", library_id)
        immich.scan_library(library_id)
        if not immich.wait_idle(timeout=IMMICH_WAIT_SECONDS):
            log.warning("Immich 任务队列 30 分钟内未空闲，继续同步已入库的资产")

        organized = cfg.paths.organized
        cache: dict[str, dict | None] = {}
        assets: dict[int, str] = {}  # main file id -> asset id (kept files only)
        for p in placements:
            if p.dst is None or p.bucket != "keep" or p.rec.id != p.main.id:
                continue
            asset = self._find_asset(immich, p.dst, organized, cache)
            if asset and asset.get("id"):
                assets[p.rec.id] = str(asset["id"])
        missing = sum(1 for p in placements if p.bucket == "keep" and p.rec.id == p.main.id and p.dst
                      and p.rec.id not in assets)
        if missing:
            log.warning("Immich 尚未入库 %d 个文件，下次运行时再同步", missing)

        self._immich_albums(ctx, events, assets)
        self._immich_metadata(ctx, placements, analyses, assets, cache)

    def _immich_albums(self, ctx: Context, events: dict[int, dict[str, Any]], assets: dict[int, str]) -> None:
        immich, state = ctx.immich, ctx.state
        assert immich is not None
        seen: set[int] = set()
        for group in events.values():
            if group["id"] in seen:
                continue
            seen.add(group["id"])
            ids = [assets[f] for f in group["members"] if f in assets]
            name = group.get("name")
            if not ids or not name:
                continue
            month = (group.get("started_at") or "")[:7]
            album = f"{month} {name}".strip()
            sig = hashlib.sha1("\n".join(sorted(ids)).encode()).hexdigest()
            if state.get(f"immich_album_sig:{album}") == sig:
                continue
            try:
                album_id = state.get(f"immich_album:{album}")
                if album_id:
                    ctx.undo.record("immich_add_assets", meta={"album_id": album_id, "asset_ids": ids})
                    immich.add_to_album(album_id, ids)
                else:
                    album_id = immich.create_album(album, ids)  # the id exists only after creation
                    ctx.undo.record("immich_album", meta={"album_id": album_id, "name": album})
                    state.set(f"immich_album:{album}", album_id)
                state.set(f"immich_album_sig:{album}", sig)
            except Exception:
                log.exception("Immich 相册同步失败: %s", album)

    def _immich_metadata(
        self, ctx: Context, placements: list[_Placement], analyses: dict[int, dict[str, Any]],
        assets: dict[int, str], cache: dict[str, dict | None],
    ) -> None:
        immich, state = ctx.immich, ctx.state
        assert immich is not None
        tag_assets: dict[str, list[str]] = {}
        for p in placements:
            asset_id = assets.get(p.rec.id)
            analysis = analyses.get(p.rec.id, {})
            if asset_id is None or p.dst is None or not analysis.get("pick"):
                continue
            desc = state.description(p.rec.id) or {}
            caption = desc.get("caption") or ""
            tags = sorted({t[0] for t in state.tags(p.rec.id)})
            rating = rating_for(analysis)
            sig = hashlib.sha1(f"{caption}|{','.join(tags)}|{rating}".encode()).hexdigest()
            if state.get(f"immich_sync:{p.rec.id}") == sig:
                continue
            try:
                if caption or rating:
                    previous = ((cache.get(str(p.dst)) or {}).get("exifInfo") or {}).get("description")
                    ctx.undo.record("immich_description", dst=p.dst, meta={"asset_id": asset_id, "previous": previous})
                    immich.update_asset(asset_id, description=caption or None, rating=rating or None)
                for tag in tags:
                    tag_assets.setdefault(tag, []).append(asset_id)
                state.set(f"immich_sync:{p.rec.id}", sig)
            except Exception:
                log.exception("Immich 描述写入失败: %s", p.rec.path)
        for tag, ids in sorted(tag_assets.items()):
            try:
                tag_id = immich.upsert_tag(f"Afterglow/{tag}")
                ctx.undo.record("immich_tag", meta={"tag_id": tag_id, "asset_ids": ids, "tag": tag})
                immich.tag_assets(tag_id, ids)
            except Exception:
                log.exception("Immich 标签写入失败: %s", tag)


STAGE = Publish
