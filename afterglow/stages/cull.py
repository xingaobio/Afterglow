"""Stage 3 - sharpness / faces / exposure / bursts, quality score and picks (ARCHITECTURE.md §5 ③)."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from PIL import Image

from afterglow.config import IMAGE_KINDS
from afterglow.stages.base import FileRecord, Stage, StepOutcome
from afterglow.util import exif
from afterglow.util.images import exposure_score, laplacian_sharpness, to_rgb_array

if TYPE_CHECKING:
    from afterglow.context import Context

log = logging.getLogger(__name__)

ANALYZE_CHUNK = 300


@dataclass(eq=False)
class _Item:
    rec: FileRecord
    sharpness: float = 0.0
    faces: int = 0
    eyes_closed: int = 0
    blurry: bool = False
    exposure: float = 1.0
    quality: float = 0.0
    tiny: bool = False
    burst_id: int | None = None
    burst_best: bool = False
    burst_alt: bool = False
    from_lumen: bool = False
    fields: dict[str, Any] = field(default_factory=dict)


def _photo_path(entry: Any) -> str | None:
    if isinstance(entry, str):
        return entry
    if isinstance(entry, dict):
        value = entry.get("path") or entry.get("file") or entry.get("source")
        return str(value) if value else None
    return None


def quality_score(sharpness: float, exposure: float, faces: int, eyes_closed: int) -> float:
    """0.6*sharpness + 25*exposure + 15*(face bonus), clipped to 0-100 (ARCHITECTURE.md §5 ③)."""
    bonus = 1.0 if (faces > 0 and eyes_closed == 0) or faces == 0 else 0.0
    return max(0.0, min(100.0, 0.6 * sharpness + 25.0 * exposure + 15.0 * bonus))


class Cull(Stage):
    name = "cull"
    requires = ("index",)
    uses_gpu = False
    per_file = True

    def run(self, ctx: Context, batch: list[FileRecord]) -> list[StepOutcome]:
        outcomes: dict[int, StepOutcome] = {}
        items: list[_Item] = []
        for rec in batch:
            analysis = rec.analysis or ctx.state.analysis(rec.id) or {}
            if rec.kind not in IMAGE_KINDS:
                outcomes[rec.id] = StepOutcome.skipped(rec.id, "not an image")
            elif rec.pair_id is not None:
                outcomes[rec.id] = StepOutcome.skipped(rec.id, "paired with RAW")
            elif analysis.get("reject_reason") == "duplicate":
                outcomes[rec.id] = StepOutcome.skipped(rec.id, "duplicate")
            else:
                items.append(_Item(rec))

        lumen_result = self._analyze_with_lumen(ctx, items)
        for item in items:
            try:
                self._measure(ctx, item, lumen_result.get(item.rec.path))
            except Exception as exc:
                log.exception("选片分析失败: %s", item.rec.path)
                outcomes[item.rec.id] = StepOutcome.failed(item.rec.id, exc)
        good = [i for i in items if i.rec.id not in outcomes]

        self._bursts(ctx, good, lumen_result.get("__bursts__", []))
        for item in good:
            try:
                self._store(ctx, item)
                outcomes[item.rec.id] = StepOutcome.done(item.rec.id, quality=round(item.quality, 1))
            except Exception as exc:
                log.exception("保存选片结果失败: %s", item.rec.path)
                outcomes[item.rec.id] = StepOutcome.failed(item.rec.id, exc)
        return [outcomes[r.id] for r in batch]

    # ------------------------------------------------------------------ analysis
    def _analyze_with_lumen(self, ctx: Context, items: list[_Item]) -> dict[str, Any]:
        """{path: photo dict} from LUMEN's analyze plus '__bursts__'; {} when LUMEN is unavailable."""
        result: dict[str, Any] = {"__bursts__": []}
        if not items:
            return result
        paths = [i.rec.path for i in items]
        try:
            for start in range(0, len(paths), ANALYZE_CHUNK):
                chunk = paths[start : start + ANALYZE_CHUNK]
                data = ctx.lumen.analyze(chunk)
                for photo in data.get("photos", []):
                    p = _photo_path(photo)
                    if p and isinstance(photo, dict):
                        result[p] = photo
                result["__bursts__"].extend(data.get("bursts", []))
        except Exception:
            log.warning("LUMEN 分析不可用，改用拉普拉斯清晰度回退", exc_info=True)
            return {"__bursts__": []}
        return result

    def _measure(self, ctx: Context, item: _Item, photo: dict[str, Any] | None) -> None:
        rec, cfg = item.rec, ctx.config.cull
        if rec.width and rec.height and min(rec.width, rec.height) < cfg.min_long_edge:
            item.tiny = True
        rgb = None
        if rec.preview_path and Path(rec.preview_path).exists():
            with Image.open(rec.preview_path) as im:
                rgb = to_rgb_array(im)
        if photo is not None and photo.get("sharpness") is not None:
            item.from_lumen = True
            item.sharpness = float(photo["sharpness"])
            item.faces = int(photo.get("faces") or 0)
            item.eyes_closed = int(photo.get("eyes_closed") or 0)
            item.blurry = bool(photo.get("blurry", item.sharpness < cfg.sharpness_min))
        else:
            if rgb is None:
                raise FileNotFoundError(f"没有预览图可用于分析: {rec.path}")
            item.sharpness = laplacian_sharpness(rgb)
            item.blurry = item.sharpness < cfg.sharpness_min
        item.exposure = exposure_score(rgb) if rgb is not None else 1.0
        item.quality = quality_score(item.sharpness, item.exposure, item.faces, item.eyes_closed)

    # ------------------------------------------------------------------ bursts
    def _bursts(self, ctx: Context, items: list[_Item], lumen_bursts: list[Any]) -> None:
        cfg = ctx.config.cull
        by_path = {i.rec.path: i for i in items if not i.tiny}
        assigned: set[int] = set()
        clusters: list[tuple[list[_Item], _Item | None]] = []
        for burst in lumen_bursts:
            if not isinstance(burst, dict):
                continue
            members = [by_path[p] for p in (_photo_path(x) for x in burst.get("photos", [])) if p in by_path]
            members = [m for m in members if m.rec.id not in assigned]
            if len(members) < 2:
                continue
            best_path = _photo_path(burst.get("best"))
            best = by_path.get(best_path) if best_path else None
            assigned.update(m.rec.id for m in members)
            clusters.append((members, best if best in members else None))
        rest = sorted((i for i in by_path.values() if i.rec.id not in assigned), key=lambda i: (
            i.rec.camera or "", i.rec.captured_at or ""))
        current: list[_Item] = []
        last_dt = None
        for item in rest:
            dt = exif.parse_iso(item.rec.captured_at)
            same = (
                current and dt is not None and last_dt is not None and current[-1].rec.camera == item.rec.camera
                and 0 <= (dt - last_dt).total_seconds() <= cfg.burst_gap_seconds
            )
            if not same:
                if len(current) >= 2:
                    clusters.append((current, None))
                current = []
            current.append(item)
            last_dt = dt
        if len(current) >= 2:
            clusters.append((current, None))

        for members, lumen_best in clusters:
            members.sort(key=lambda i: (i.rec.captured_at or "", i.rec.id))
            ranked = sorted(members, key=lambda i: i.quality, reverse=True)
            keep: list[_Item] = []
            if lumen_best is not None:
                keep.append(lumen_best)
            for cand in ranked:
                if len(keep) >= cfg.keep_per_burst:
                    break
                if cand not in keep:
                    keep.append(cand)
            gid = ctx.state.create_group(
                "burst", [m.rec.id for m in members],
                meta={"best": [k.rec.id for k in keep], "count": len(members),
                      "source": "lumen" if lumen_best else "gap"},
            )
            for m in members:
                m.burst_id = gid
                m.burst_best = m in keep
                m.burst_alt = m not in keep

    # ------------------------------------------------------------------ result
    def _store(self, ctx: Context, item: _Item) -> None:
        cfg = ctx.config.cull
        reason: str | None = None
        if item.tiny:
            reason = "tiny"
        elif item.burst_alt:
            reason = "burst_alt"
        elif item.sharpness < cfg.sharpness_min:
            reason = "blurry"
        elif cfg.reject_eyes_closed and item.eyes_closed > 0 and item.faces > 0:
            reason = "eyes_closed"
        pick = reason is None
        ctx.state.set_analysis(
            item.rec.id,
            sharpness=round(item.sharpness, 2), faces=item.faces, eyes_closed=item.eyes_closed,
            blurry=int(item.blurry), exposure_score=round(item.exposure, 4), quality=round(item.quality, 2),
            burst_id=item.burst_id, burst_best=int(item.burst_best), pick=int(pick), reject_reason=reason,
        )


STAGE = Cull
