"""Stage 7 - HDR / focus-stack / panorama merging (ARCHITECTURE.md §5 ⑦)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np
from PIL import Image

from afterglow.stages.base import FileRecord, Stage, StepOutcome
from afterglow.stages.understand import sanitize_name
from afterglow.util.images import from_rgb_array, save_jpeg, to_rgb_array

if TYPE_CHECKING:
    from afterglow.context import Context

log = logging.getLogger(__name__)

MERGE_KINDS = ("hdr", "focus", "pano")
FALLBACK_MAX_EDGE = 3000
PYRAMID_LEVELS = 5
SCATTERED = "零散"


def _resize_to(rgb: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    h, w = size
    if rgb.shape[:2] == (h, w):
        return rgb
    return cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)


def _limit(rgb: np.ndarray, max_edge: int = FALLBACK_MAX_EDGE) -> np.ndarray:
    scale = max_edge / max(rgb.shape[:2])
    if scale >= 1:
        return rgb
    h, w = rgb.shape[:2]
    return cv2.resize(rgb, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)


def merge_hdr(images: list[np.ndarray]) -> np.ndarray:
    """Mertens exposure fusion (after MTB alignment when possible)."""
    base = images[0].shape[:2]
    bgr = [cv2.cvtColor(_resize_to(im, base), cv2.COLOR_RGB2BGR) for im in images]
    try:
        cv2.createAlignMTB().process(bgr, bgr)
    except cv2.error:
        log.warning("MTB 对齐失败，直接融合")
    fused = cv2.createMergeMertens().process(bgr)
    out = np.clip(fused * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


def _laplacian_pyramid(img: np.ndarray, levels: int) -> list[np.ndarray]:
    gauss = [img]
    for _ in range(levels):
        gauss.append(cv2.pyrDown(gauss[-1]))
    pyr = []
    for i in range(levels):
        up = cv2.pyrUp(gauss[i + 1], dstsize=(gauss[i].shape[1], gauss[i].shape[0]))
        pyr.append(gauss[i] - up)
    pyr.append(gauss[-1])
    return pyr


def merge_focus(images: list[np.ndarray]) -> np.ndarray:
    """Focus stacking: at every Laplacian-pyramid level keep the coefficient with the largest magnitude."""
    base = images[0].shape[:2]
    pyramids = [_laplacian_pyramid(_resize_to(im, base).astype(np.float32), PYRAMID_LEVELS) for im in images]
    fused: list[np.ndarray] = []
    for level in range(PYRAMID_LEVELS):
        stack = np.stack([p[level] for p in pyramids])  # (n, h, w, 3)
        strength = np.abs(stack).sum(axis=-1)
        pick = np.argmax(strength, axis=0)
        fused.append(np.take_along_axis(stack, pick[None, ..., None], axis=0)[0])
    fused.append(np.mean([p[-1] for p in pyramids], axis=0))
    out = fused[-1]
    for level in range(PYRAMID_LEVELS - 1, -1, -1):
        out = cv2.pyrUp(out, dstsize=(fused[level].shape[1], fused[level].shape[0])) + fused[level]
    return np.clip(out, 0, 255).astype(np.uint8)


def merge_pano(images: list[np.ndarray]) -> np.ndarray:
    stitcher = cv2.Stitcher.create(cv2.Stitcher_PANORAMA)
    status, pano = stitcher.stitch([cv2.cvtColor(im, cv2.COLOR_RGB2BGR) for im in images])
    if status != cv2.Stitcher_OK or pano is None:
        raise RuntimeError(f"全景拼接失败 (OpenCV 状态码 {status})")
    return cv2.cvtColor(pano, cv2.COLOR_BGR2RGB)


OPENCV_MERGERS = {"hdr": merge_hdr, "focus": merge_focus, "pano": merge_pano}


class Merge(Stage):
    name = "merge"
    requires = ("understand",)
    uses_gpu = True
    per_file = False

    def run(self, ctx: Context, batch: list[FileRecord]) -> list[StepOutcome]:
        state = ctx.state
        event_of: dict[int, str] = {}
        for g in state.groups("event"):
            for fid in g["members"]:
                event_of[fid] = g.get("name") or SCATTERED
        status: dict[int, StepOutcome] = {}
        merged = 0
        for kind in MERGE_KINDS:
            for group in state.groups(kind):
                if any(Path(o["path"]).exists() for o in state.outputs(group_id=group["id"])):
                    outcome = [StepOutcome.done(fid) for fid in group["members"]]
                else:
                    try:
                        out = self._merge_group(ctx, kind, group, event_of)
                        merged += 1
                        outcome = [StepOutcome.done(fid, output=str(out)) for fid in group["members"]]
                    except Exception as exc:
                        log.exception("合成失败: %s group %s", kind, group["id"])
                        outcome = [StepOutcome.failed(fid, exc) for fid in group["members"]]
                for o in outcome:
                    if status.get(o.file_id) is None or o.status == "failed":
                        status[o.file_id] = o
        for rec in state.files_where(missing=False):
            status.setdefault(rec.id, StepOutcome.skipped(rec.id, "not in a mergeable group"))
        state.set_steps(list(status.values()), self.name)
        log.info("合成完成: 新生成 %d 张", merged)
        return []

    def _merge_group(self, ctx: Context, kind: str, group: dict[str, Any], event_of: dict[int, str]) -> Path:
        state, cfg = ctx.state, ctx.config
        recs = [r for r in (state.get_file(fid) for fid in group["members"]) if r is not None]
        if len(recs) < 2:
            raise ValueError("成员不足")
        arrays = [self._member_image(ctx, r) for r in recs]
        lib = ctx.lumen_lib
        engine = "lumen"
        if lib is not None and getattr(lib, "available", False):
            result = lib.merge_images(arrays, kind)
        else:
            engine = "opencv"
            result = OPENCV_MERGERS[kind]([_limit(a) for a in arrays])
        event = sanitize_name(event_of.get(recs[0].id, SCATTERED), 40) or SCATTERED
        out = cfg.paths.output / "merged" / event / f"{kind}_{Path(recs[0].rel_path).stem}.jpg"
        save_jpeg(from_rgb_array(np.asarray(result)), out, cfg.lumen.export_quality)
        state.add_output(kind, str(out), group_id=group["id"],
                         params={"engine": engine, "members": [r.id for r in recs]})
        return out

    def _member_image(self, ctx: Context, rec: FileRecord) -> np.ndarray:
        """The member's process export when present, else an on-the-fly LUMEN export into cache, else its preview."""
        state = ctx.state
        for o in reversed(state.outputs(file_id=rec.id, kind="export")):
            if Path(o["path"]).exists():
                return self._read(o["path"])
        cache = ctx.config.paths.cache / "merge" / f"{rec.id}_{Path(rec.rel_path).stem}.jpg"
        if not cache.exists():
            try:
                cache.parent.mkdir(parents=True, exist_ok=True)
                ctx.lumen.open(rec.path)
                ctx.lumen.auto_tone(rec.path)
                ctx.lumen.export(rec.path, str(cache), ctx.config.lumen.export_quality, None)
            except Exception:
                log.warning("现场导出失败，改用预览图: %s", rec.path, exc_info=True)
        if cache.exists():
            return self._read(cache)
        if rec.preview_path and Path(rec.preview_path).exists():
            return self._read(rec.preview_path)
        raise FileNotFoundError(f"没有可用于合成的图像: {rec.path}")

    @staticmethod
    def _read(path: str | Path) -> np.ndarray:
        with Image.open(path) as im:
            return to_rgb_array(im)


STAGE = Merge
