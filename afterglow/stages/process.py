"""Stage 6 - LUMEN export, straightening, crops and AI enhancement (ARCHITECTURE.md §5 ⑥)."""

from __future__ import annotations

import logging
from contextlib import nullcontext
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from PIL import Image

from afterglow.config import IMAGE_KINDS
from afterglow.integrations.llm import LLMClient
from afterglow.stages.base import FileRecord, Stage, StepOutcome
from afterglow.stages.understand import resized_jpeg
from afterglow.util.images import (
    crop_ratio,
    from_rgb_array,
    rotate_and_crop,
    saliency_center,
    save_jpeg,
    straighten_estimate,
    to_rgb_array,
)

if TYPE_CHECKING:
    from afterglow.context import Context

log = logging.getLogger(__name__)

#: LUMEN slider ranges (lumen/commands.py CHANGES): exposure in EV, everything else -100…100.
SLIDER_RANGE: dict[str, tuple[float, float]] = {"exposure": (-5.0, 5.0), "*": (-100.0, 100.0)}

DENOISE_ISO = 3200
SMALL_LONG_EDGE = 3000
DENOISE_AMOUNT = 70
MIN_ROTATION_DEG = 0.05


def export_path(ctx: Context, rec: FileRecord) -> Path:
    """output/exports/<rel dir>/<stem>.<export_format> for a source file."""
    ext = ctx.config.lumen.export_format.lower().lstrip(".")
    ext = "jpg" if ext == "jpeg" else ext
    rel = Path(rec.rel_path)
    return ctx.config.paths.output / "exports" / rel.parent / f"{rel.stem}.{ext}"


class Process(Stage):
    name = "process"
    requires = ("understand",)
    uses_gpu = True
    per_file = True

    def run(self, ctx: Context, batch: list[FileRecord]) -> list[StepOutcome]:
        cfg = ctx.config
        outcomes: list[StepOutcome] = []
        for rec in batch:
            analysis = rec.analysis or ctx.state.analysis(rec.id) or {}
            if rec.kind not in IMAGE_KINDS:
                outcomes.append(StepOutcome.skipped(rec.id, "not an image"))
            elif rec.pair_id is not None:
                outcomes.append(StepOutcome.skipped(rec.id, "paired with RAW"))
            elif cfg.process.scope == "picks" and not analysis.get("pick"):
                outcomes.append(StepOutcome.skipped(rec.id, "not a pick"))
            elif rec.kind != "raw" and cfg.process.scope != "all":
                outcomes.append(StepOutcome.skipped(rec.id, "only RAW files are exported in picks scope"))
            else:
                try:
                    outcomes.append(StepOutcome.done(rec.id, **self._process_one(ctx, rec)))
                except Exception as exc:
                    log.exception("处理失败: %s", rec.path)
                    outcomes.append(StepOutcome.failed(rec.id, exc))
        return outcomes

    def _process_one(self, ctx: Context, rec: FileRecord) -> dict[str, Any]:
        cfg = ctx.config
        detail: dict[str, Any] = {}
        exported = self._export(ctx, rec)
        with Image.open(exported) as im:
            rgb = to_rgb_array(im)
        quality = cfg.lumen.export_quality

        if cfg.process.straighten:
            rgb, info = self._straighten(ctx, rec, exported, rgb)
            if info:
                detail["straighten"] = info

        crops = self._crops(ctx, rec, rgb)
        if crops:
            detail["crops"] = crops

        detail["enhance"] = self._enhance(ctx, rec, rgb, quality)
        return detail

    def _export(self, ctx: Context, rec: FileRecord) -> Path:
        cfg = ctx.config
        out = export_path(ctx, rec)
        out.parent.mkdir(parents=True, exist_ok=True)
        lumen = ctx.lumen
        description = ctx.state.description(rec.id) or {}
        style = description.get("style_preset") or "default"
        presets = cfg.process.style_presets
        preset = presets.get(style) or presets.get("default") or ""
        grade: dict[str, Any] | None = None
        # Edits live in the lumen-cli subprocess; a restart between calls would drop them, so the
        # whole open -> auto -> preset -> grade -> export chain runs inside one sequence.
        sequence = getattr(lumen, "sequence", None)
        with sequence() if callable(sequence) else nullcontext():
            lumen.open(rec.path)
            lumen.auto_tone(rec.path)
            if preset:
                lumen.apply_preset(rec.path, preset)
            grade = self._grade(ctx, rec, description)
            lumen.export(rec.path, str(out), cfg.lumen.export_quality, cfg.lumen.export_long_edge or None)
        if not out.exists():
            raise FileNotFoundError(f"LUMEN 没有写出导出文件: {out}")
        ctx.state.add_output(
            "export", str(out), file_id=rec.id, params={"preset": preset or None, "style": style, "grade": grade},
        )
        return out

    def _grade(self, ctx: Context, rec: FileRecord, description: dict[str, Any]) -> dict[str, Any] | None:
        """Natural-language grading (§11.2) on top of the automatic adjustment; None when not applied.

        LUMEN's edit.apply takes *final* slider values, not increments.  So the model looks at LUMEN's
        own rendering of the photo as it stands (after auto tone / preset), proposes small deltas, and
        those deltas - scaled by grade.strength - are added to the current values before being applied.
        Any failure leaves the automatic result untouched.
        """
        cfg = ctx.config
        lumen = ctx.lumen
        grade_fn = getattr(ctx.llm, "grade", None)
        get_edits = getattr(lumen, "get_edits", None)
        scene = description.get("scene")
        if (
            cfg.grade.mode != "llm"
            or cfg.grade.strength <= 0
            or not callable(grade_fn)
            or not callable(get_edits)
            or not scene
            or scene not in cfg.grade.scenes
        ):
            return None
        try:
            current = ((get_edits(rec.path) or {}).get("state") or {}).get("adjustments")
            if not isinstance(current, dict):
                raise ValueError("LUMEN did not report the current adjustments")
            image = self._graded_view(ctx, rec)
            result = grade_fn(image, scene, description.get("caption") or "", cfg.grade.instruction, cfg.llm.language)
            # the client already normalises; re-clamp so any LLM double (or future client) is safe too
            deltas = LLMClient.normalise_grade((result or {}).get("adjustments"))
            if not deltas:
                return None
            scale = cfg.grade.strength / 100.0
            final: dict[str, float] = {}
            for key, delta in deltas.items():
                base = current.get(key)
                base = float(base) if isinstance(base, int | float) and not isinstance(base, bool) else 0.0
                low, high = SLIDER_RANGE.get(key, SLIDER_RANGE["*"])
                value = max(low, min(high, base + delta * scale))
                final[key] = round(value, 2) if key == "exposure" else round(value)
            lumen.apply_edits(rec.path, {"adjustments": final}, amount=100, label="afterglow grade")
            return {
                "adjustments": deltas, "applied": final, "reason": str((result or {}).get("reason") or ""),
                "strength": cfg.grade.strength,
            }
        except Exception as exc:
            log.warning("自然语言调色未应用，保留自动调色结果: %s (%s: %s)", rec.path, type(exc).__name__, exc)
            log.debug("grade traceback", exc_info=True)
            return None

    @staticmethod
    def _graded_view(ctx: Context, rec: FileRecord) -> bytes:
        """What the model should judge: LUMEN's rendering with the edits so far; else the camera preview."""
        edge = ctx.config.llm.max_image_edge
        render = getattr(ctx.lumen, "render_preview", None)
        if callable(render):
            try:
                data = render(rec.path, edge)
                if data:
                    return data
            except Exception as exc:
                log.debug("render_preview failed, using the stored preview: %s", exc)
        if not rec.preview_path:
            raise FileNotFoundError("no preview image for grading")
        return resized_jpeg(rec.preview_path, edge)

    def _straighten(
        self, ctx: Context, rec: FileRecord, exported: Path, rgb: np.ndarray
    ) -> tuple[np.ndarray, dict[str, Any] | None]:
        cfg = ctx.config
        lib = ctx.lumen_lib
        if lib is not None and getattr(lib, "available", False):
            angle, confidence = lib.straighten(rgb)
        else:
            angle, confidence = straighten_estimate(rgb)
        if (
            angle is None
            or confidence < cfg.process.straighten_min_confidence
            or abs(angle) > cfg.process.straighten_max_degrees
            or abs(angle) < MIN_ROTATION_DEG
        ):
            return rgb, None
        rotated = rotate_and_crop(rgb, angle)
        save_jpeg(from_rgb_array(rotated), exported, cfg.lumen.export_quality)
        info = {"angle": angle, "confidence": confidence}
        ctx.state.add_output("straighten", str(exported), file_id=rec.id, params=info)
        return rotated, info

    def _crops(self, ctx: Context, rec: FileRecord, rgb: np.ndarray) -> list[str]:
        ratios = ctx.config.process.crops
        if not ratios:
            return []
        center = saliency_center(rgb)
        rel = Path(rec.rel_path)
        done: list[str] = []
        for ratio in ratios:
            out = ctx.config.paths.output / "crops" / rel.parent / f"{rel.stem}_{ratio.replace(':', 'x')}.jpg"
            save_jpeg(from_rgb_array(crop_ratio(rgb, ratio, center)), out, ctx.config.lumen.export_quality)
            ctx.state.add_output("crop", str(out), file_id=rec.id, params={"ratio": ratio, "center": list(center)})
            done.append(ratio)
        return done

    def _enhance(self, ctx: Context, rec: FileRecord, rgb: np.ndarray, quality: int) -> str:
        cfg = ctx.config
        mode = cfg.lumen.enhance
        if mode == "off":
            return "off"
        long_edge = max(rec.width or 0, rec.height or 0) or max(rgb.shape[:2])
        high_iso = rec.iso is not None and rec.iso >= DENOISE_ISO
        small = long_edge < SMALL_LONG_EDGE
        if mode == "auto" and not (high_iso or small):
            return "not needed"
        lib = ctx.lumen_lib
        if lib is None or not getattr(lib, "available", False):
            log.warning("LumenLib 不可用，跳过 AI 增强: %s", rec.path)
            return "skipped"
        result = rgb
        steps: list[str] = []
        if mode == "all" or high_iso:
            result = lib.denoise(result, DENOISE_AMOUNT)
            steps.append("denoise")
        if mode == "all" or small:
            result = lib.super_resolution(result, cfg.lumen.enhance_scale)
            steps.append("super_resolution")
        rel = Path(rec.rel_path)
        suffix = f"x{cfg.lumen.enhance_scale}" if "super_resolution" in steps else "dn"
        out = cfg.paths.output / "enhanced" / rel.parent / f"{rel.stem}_{suffix}.jpg"
        save_jpeg(from_rgb_array(np.asarray(result)), out, quality)
        ctx.state.add_output("enhance", str(out), file_id=rec.id, params={"steps": steps})
        return "+".join(steps)


STAGE = Process
