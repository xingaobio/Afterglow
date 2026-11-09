"""Stage 4 - events and HDR / focus / pano / timelapse sequence detection (ARCHITECTURE.md §5 ④).

Global stage: it only reads `files` / `analysis`, creates `groups`, and marks every file's `group` step done.
Files already belonging to a group of the same family (event, or one of the sequence kinds) are left alone, so a
rerun after new files arrive only groups the new ones.
"""

from __future__ import annotations

import logging
import math
import statistics
from collections.abc import Callable
from datetime import datetime
from typing import TYPE_CHECKING, Any

from afterglow.stages.base import FileRecord, Stage, StepOutcome
from afterglow.util import exif
from afterglow.util.hashing import hamming

if TYPE_CHECKING:
    from afterglow.context import Context

log = logging.getLogger(__name__)

SEQUENCE_KINDS = ("hdr", "focus", "pano", "timelapse")
SCATTERED = "零散"

HDR_GAP, HDR_MIN, HDR_MAX, HDR_SPAN_EV = 3.0, 3, 9, 2.0
FOCUS_GAP, FOCUS_MIN, FOCUS_DHASH_MAX = 5.0, 4, 6
PANO_GAP, PANO_MIN, PANO_MAX, PANO_DHASH = 10.0, 3, 20, (12, 30)
TIMELAPSE_CV_MAX, TIMELAPSE_MIN_INTERVAL, TIMELAPSE_MAX_GAP = 0.2, 1.0, 3600.0


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2
    return 2 * 6371.0 * math.asin(min(1.0, math.sqrt(a)))


class _Item:
    """A file with its parsed capture time."""

    __slots__ = ("dt", "rec")

    def __init__(self, rec: FileRecord, dt: datetime) -> None:
        self.rec = rec
        self.dt = dt


def _gap(a: _Item, b: _Item) -> float:
    return (b.dt - a.dt).total_seconds()


def _split_runs(items: list[_Item], max_gap: float, same_camera: bool = True) -> list[list[_Item]]:
    """Maximal runs of time-sorted items whose neighbours are <= max_gap apart (and from the same body)."""
    runs: list[list[_Item]] = []
    cur: list[_Item] = []
    for item in items:
        if cur and 0 <= _gap(cur[-1], item) <= max_gap and (not same_camera or cur[-1].rec.camera == item.rec.camera):
            cur.append(item)
        else:
            if cur:
                runs.append(cur)
            cur = [item]
    if cur:
        runs.append(cur)
    return runs


def _monotone_windows(vals: list[float], min_step: float) -> list[tuple[int, int]]:
    """Greedy maximal strictly-monotone windows [start, end) with step >= min_step and at least 2 elements."""
    windows: list[tuple[int, int]] = []
    i, n = 0, len(vals)
    while i < n - 1:
        d = vals[i + 1] - vals[i]
        if abs(d) < min_step:
            i += 1
            continue
        sign = 1.0 if d > 0 else -1.0
        j = i + 1
        while j + 1 < n and (vals[j + 1] - vals[j]) * sign >= min_step:
            j += 1
        windows.append((i, j + 1))
        i = j + 1
    return windows


def _same(a: float | None, b: float | None) -> bool:
    if a is None or b is None:
        return a is b
    return math.isclose(a, b, rel_tol=1e-3, abs_tol=1e-9)


class Group(Stage):
    name = "group"
    requires = ("cull",)
    uses_gpu = False
    per_file = False

    def run(self, ctx: Context, batch: list[FileRecord]) -> list[StepOutcome]:
        state = ctx.state
        files = state.files_where(missing=False)
        grouped: dict[str, set[int]] = {"event": set(), "sequence": set()}
        for g in state.groups("event"):
            grouped["event"].update(g["members"])
        for kind in SEQUENCE_KINDS:
            for g in state.groups(kind):
                grouped["sequence"].update(g["members"])

        items: list[_Item] = []
        for rec in files:
            dt = exif.parse_iso(rec.captured_at)
            if dt is None or not (rec.is_image or rec.is_video) or rec.pair_id is not None:
                continue
            if (rec.analysis or {}).get("reject_reason") == "duplicate":
                continue
            items.append(_Item(rec, dt))
        items.sort(key=lambda i: (i.dt, i.rec.id))

        created = {"event": self._events(ctx, [i for i in items if i.rec.id not in grouped["event"]])}
        images = [i for i in items if i.rec.is_image and i.rec.id not in grouped["sequence"]]
        claimed: set[int] = set()
        for kind, detector in (
            ("hdr", self._hdr), ("timelapse", self._timelapse), ("focus", self._focus), ("pano", self._pano)
        ):
            free = [i for i in images if i.rec.id not in claimed]
            found = detector(ctx, free)
            for members, meta in found:
                state.create_group(kind, [m.rec.id for m in members], meta=meta)
                claimed.update(m.rec.id for m in members)
            created[kind] = len(found)
        log.info("分组完成: %s", ", ".join(f"{k} {v}" for k, v in created.items()))
        state.set_steps([StepOutcome.done(f.id) for f in files], self.name)
        return []

    # ------------------------------------------------------------------ events
    def _events(self, ctx: Context, items: list[_Item]) -> int:
        cfg = ctx.config.events
        segments: list[list[_Item]] = []
        cur: list[_Item] = []
        last_gps: tuple[float, float] | None = None
        for item in items:
            has_gps = item.rec.gps_lat is not None and item.rec.gps_lon is not None
            gps = (item.rec.gps_lat, item.rec.gps_lon) if has_gps else None
            split = False
            if cur:
                if _gap(cur[-1], item) > cfg.gap_hours * 3600:
                    split = True
                elif gps and last_gps and haversine_km(*last_gps, *gps) > cfg.gps_km:
                    split = True
            if split:
                segments.append(cur)
                cur = []
                last_gps = None
            cur.append(item)
            if gps:
                last_gps = gps
        if cur:
            segments.append(cur)

        # Incremental runs: photos that arrive later join the event they belong to instead of
        # spawning a second one (same for the per-month scattered bucket).
        existing = ctx.state.groups("event")
        extend = getattr(ctx.state, "add_group_members", None)
        month_groups = {g["meta"].get("month"): g for g in existing if g["meta"].get("scattered")}
        spans = []
        for g in existing:
            start, end = exif.parse_iso(g.get("started_at")), exif.parse_iso(g.get("ended_at"))
            if start and end and not g["meta"].get("scattered"):
                spans.append((start, end, g))
        gap = cfg.gap_hours * 3600

        def neighbour(seg: list[_Item]) -> dict[str, Any] | None:
            first, last = seg[0].dt, seg[-1].dt
            for start, end, g in spans:
                if (first - end).total_seconds() <= gap and (start - last).total_seconds() <= gap:
                    return g
            return None

        count = 0
        scattered: dict[str, list[_Item]] = {}
        for seg in segments:
            near = neighbour(seg) if callable(extend) else None
            if near is not None:
                extend(near["id"], [i.rec.id for i in seg])
                continue
            if len(seg) < cfg.min_photos:
                for item in seg:
                    scattered.setdefault(item.dt.strftime("%Y-%m"), []).append(item)
                continue
            ctx.state.create_group("event", [i.rec.id for i in seg], meta=self._event_meta(seg))
            count += 1
        for month, seg in sorted(scattered.items()):
            seg.sort(key=lambda i: (i.dt, i.rec.id))
            if month in month_groups and callable(extend):
                extend(month_groups[month]["id"], [i.rec.id for i in seg])
                continue
            meta = self._event_meta(seg)
            meta.update(scattered=True, month=month)
            ctx.state.create_group("event", [i.rec.id for i in seg], name=SCATTERED, meta=meta)
            count += 1
        return count

    @staticmethod
    def _event_meta(seg: list[_Item]) -> dict[str, Any]:
        lats = [i.rec.gps_lat for i in seg if i.rec.gps_lat is not None and i.rec.gps_lon is not None]
        lons = [i.rec.gps_lon for i in seg if i.rec.gps_lat is not None and i.rec.gps_lon is not None]
        center = [round(statistics.fmean(lats), 6), round(statistics.fmean(lons), 6)] if lats else None
        cameras = sorted({i.rec.camera for i in seg if i.rec.camera})
        return {"gps_center": center, "cameras": cameras, "count": len(seg)}

    # ------------------------------------------------------------------ sequences
    def _hdr(self, ctx: Context, items: list[_Item]) -> list[tuple[list[_Item], dict[str, Any]]]:
        found: list[tuple[list[_Item], dict[str, Any]]] = []
        for run in _split_runs(items, HDR_GAP):
            if len(run) < HDR_MIN:
                continue
            recs = [i.rec for i in run]
            if all(r.exposure_bias is not None for r in recs):
                mode, vals, step = "bias", [float(r.exposure_bias or 0.0) for r in recs], 0.3
            elif all(r.shutter and r.shutter > 0 for r in recs):
                mode, vals, step = "shutter", [math.log2(r.shutter or 1.0) for r in recs], 0.5
            else:
                continue
            windows = [(s, e) for s, e in _monotone_windows(vals, step) if HDR_MIN <= e - s <= HDR_MAX]
            if not windows and HDR_MIN <= len(run) <= HDR_MAX:
                # Auto-bracketing often shoots 0, -2, +2: not monotone, but every frame differs from
                # every other by at least one step and the whole run spans the required range.
                ordered = sorted(vals)
                if all(b - a >= step for a, b in zip(ordered, ordered[1:], strict=False)):
                    windows = [(0, len(run))]
            for start, end in windows:
                window = run[start:end]
                span = max(vals[start:end]) - min(vals[start:end])
                if span < HDR_SPAN_EV - 0.01:
                    continue
                if mode == "shutter" and not all(
                    _same(w.rec.aperture, window[0].rec.aperture) and _same(w.rec.iso, window[0].rec.iso)
                    for w in window
                ):
                    continue
                found.append((window, {
                    "mode": mode, "camera": window[0].rec.camera, "span_ev": round(span, 2),
                    "exposure_bias": [w.rec.exposure_bias for w in window], "shutter": [w.rec.shutter for w in window],
                }))
        return found

    def _focus(self, ctx: Context, items: list[_Item]) -> list[tuple[list[_Item], dict[str, Any]]]:
        found: list[tuple[list[_Item], dict[str, Any]]] = []
        for run in _split_runs(items, FOCUS_GAP):
            if len(run) < FOCUS_MIN:
                continue
            if all(i.rec.focus_distance is not None for i in run):
                vals = [float(i.rec.focus_distance or 0.0) for i in run]
                for start, end in _monotone_windows(vals, 1e-6):
                    if end - start >= FOCUS_MIN:
                        window = run[start:end]
                        found.append((window, {"mode": "focus_distance", "camera": window[0].rec.camera,
                                               "focus_distance": vals[start:end]}))
                continue
            for window in self._chain(run, lambda a, b: (
                a.rec.dhash and b.rec.dhash and hamming(a.rec.dhash, b.rec.dhash) <= FOCUS_DHASH_MAX
                and _same(a.rec.shutter, b.rec.shutter) and _same(a.rec.aperture, b.rec.aperture)
            )):
                if len(window) >= FOCUS_MIN:
                    found.append((window, {"mode": "dhash", "camera": window[0].rec.camera}))
        return found

    def _pano(self, ctx: Context, items: list[_Item]) -> list[tuple[list[_Item], dict[str, Any]]]:
        lo, hi = PANO_DHASH
        found: list[tuple[list[_Item], dict[str, Any]]] = []

        def overlapping(a: _Item, b: _Item) -> bool:
            if not (a.rec.dhash and b.rec.dhash):
                return False
            locked = (
                _same(a.rec.shutter, b.rec.shutter) and _same(a.rec.aperture, b.rec.aperture)
                and _same(a.rec.iso, b.rec.iso)
            )
            return locked and lo <= hamming(a.rec.dhash, b.rec.dhash) <= hi

        for run in _split_runs(items, PANO_GAP):
            for chain in self._chain(run, overlapping):
                for start in range(0, len(chain), PANO_MAX):
                    window = chain[start : start + PANO_MAX]
                    if len(window) >= PANO_MIN:
                        found.append((window, {"camera": window[0].rec.camera, "count": len(window)}))
        return found

    def _timelapse(self, ctx: Context, items: list[_Item]) -> list[tuple[list[_Item], dict[str, Any]]]:
        min_frames = ctx.config.video.timelapse_min_frames
        found: list[tuple[list[_Item], dict[str, Any]]] = []
        for run in _split_runs(items, TIMELAPSE_MAX_GAP):
            i, n = 0, len(run)
            while n - i >= min_frames:
                intervals: list[float] = []
                end = i
                for j in range(i + 1, n):
                    intervals.append(_gap(run[j - 1], run[j]))
                    mean = statistics.fmean(intervals)
                    if len(intervals) >= 2 and mean > 0 and statistics.pstdev(intervals) / mean >= TIMELAPSE_CV_MAX:
                        intervals.pop()
                        break
                    end = j
                window = run[i : end + 1]
                mean = statistics.fmean(intervals) if intervals else 0.0
                if len(window) >= min_frames and mean >= TIMELAPSE_MIN_INTERVAL:
                    found.append((window, {
                        "camera": window[0].rec.camera, "interval_mean": round(mean, 3),
                        "interval_std": round(statistics.pstdev(intervals), 3), "count": len(window),
                    }))
                    i = end + 1
                else:
                    i += 1
        return found

    @staticmethod
    def _chain(run: list[_Item], linked: Callable[[_Item, _Item], Any]) -> list[list[_Item]]:
        """Split `run` into maximal chains whose neighbours satisfy `linked`; singletons are dropped."""
        chains: list[list[_Item]] = []
        cur: list[_Item] = []
        for item in run:
            if cur and linked(cur[-1], item):
                cur.append(item)
            else:
                if len(cur) > 1:
                    chains.append(cur)
                cur = [item]
        if len(cur) > 1:
            chains.append(cur)
        return chains


STAGE = Group
