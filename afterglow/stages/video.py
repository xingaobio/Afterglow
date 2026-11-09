"""Stage 8 - ffprobe, keyframes, proxy encodes and timelapse videos (ARCHITECTURE.md §5 ⑧)."""

from __future__ import annotations

import json
import logging
import os
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any

from PIL import Image

from afterglow.stages.base import FileRecord, Stage, StepOutcome
from afterglow.stages.understand import resized_jpeg, sanitize_name
from afterglow.util import exif

if TYPE_CHECKING:
    from afterglow.context import Context

log = logging.getLogger(__name__)

PROBE_TIMEOUT = 120
FRAME_TIMEOUT = 120
ENCODE_TIMEOUT = 6 * 3600
KEYFRAME_EDGE = 1280
SCATTERED = "零散"


def _run(cmd: list[str], timeout: int) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=timeout)


class Video(Stage):
    name = "video"
    requires = ("understand",)
    uses_gpu = True
    per_file = True

    def __init__(self) -> None:
        self._encoders: set[str] | None = None  # `ffmpeg -encoders` result, probed once per stage instance
        self._timelapse_run: int | None = None

    # ------------------------------------------------------------------ encoder detection
    def available_encoders(self) -> set[str]:
        if self._encoders is None:
            try:
                proc = _run(["ffmpeg", "-hide_banner", "-encoders"], PROBE_TIMEOUT)
                names = {line.split()[1] for line in proc.stdout.splitlines() if len(line.split()) >= 2}
            except (FileNotFoundError, subprocess.TimeoutExpired):
                log.warning("ffmpeg 不可用，无法探测编码器")
                names = set()
            self._encoders = names
        return self._encoders

    def pick_encoder(self, ctx: Context) -> str:
        wanted = ctx.config.video.encoder
        if wanted in self.available_encoders():
            return wanted
        if wanted != "libx264":
            log.warning("编码器 %s 不可用，回退 libx264", wanted)
        return "libx264"

    # ------------------------------------------------------------------ stage
    def run(self, ctx: Context, batch: list[FileRecord]) -> list[StepOutcome]:
        outcomes: list[StepOutcome] = []
        for rec in batch:
            if not rec.is_video:
                outcomes.append(StepOutcome.skipped(rec.id, "not a video"))
                continue
            try:
                outcomes.append(StepOutcome.done(rec.id, **self._process_video(ctx, rec)))
            except Exception as exc:
                log.exception("视频处理失败: %s", rec.path)
                outcomes.append(StepOutcome.failed(rec.id, exc))
        if self._timelapse_run != ctx.run_id or self._timelapse_run is None:
            self._timelapse_run = ctx.run_id
            self._timelapses(ctx)
        return outcomes

    def _process_video(self, ctx: Context, rec: FileRecord) -> dict[str, Any]:
        info = self._probe(ctx, rec)
        frames = self._keyframes(ctx, rec, info.get("duration") or rec.duration or 0.0)
        detail: dict[str, Any] = {"keyframes": len(frames)}
        if frames:
            self._set_preview(ctx, rec, frames[0])
            self._tag_frames(ctx, rec, frames)
        detail["proxy"] = str(self._proxy(ctx, rec)) if not ctx.dry_run else None
        return detail

    def _probe(self, ctx: Context, rec: FileRecord) -> dict[str, Any]:
        cmd = ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", "-show_streams", rec.path]
        proc = _run(cmd, PROBE_TIMEOUT)
        if proc.returncode != 0:
            raise RuntimeError(f"ffprobe 失败 (返回码 {proc.returncode}): {(proc.stderr or '').strip()[:300]}")
        data = json.loads(proc.stdout or "{}")
        fmt = data.get("format", {})
        stream = next((s for s in data.get("streams", []) if s.get("codec_type") == "video"), {})
        update: dict[str, Any] = {}
        try:
            duration = float(fmt.get("duration") or stream.get("duration") or 0.0)
        except (TypeError, ValueError):
            duration = 0.0
        if duration > 0:
            update["duration"] = duration
        if stream.get("width") and stream.get("height"):
            update["width"], update["height"] = int(stream["width"]), int(stream["height"])
        created = exif.parse_exif_datetime((fmt.get("tags") or {}).get("creation_time"))
        if created is not None and rec.captured_src != "exif":
            update["captured_at"], update["captured_src"] = exif.fmt_iso(created), "exif"
        if update:
            ctx.state.upsert_file({"path": rec.path, **update})
        return update

    def _keyframes(self, ctx: Context, rec: FileRecord, duration: float) -> list[Path]:
        count = max(0, ctx.config.video.keyframes)
        if count == 0:
            return []
        folder = ctx.config.paths.cache / "keyframes" / str(rec.id)
        folder.mkdir(parents=True, exist_ok=True)
        frames: list[Path] = []
        for i in range(count):
            at = duration * (i + 0.5) / count if duration > 0 else 0.0
            out = folder / f"kf_{i}.jpg"
            cmd = ["ffmpeg", "-y", "-v", "error", "-ss", f"{at:.3f}", "-i", rec.path, "-frames:v", "1",
                   "-vf", f"scale='min({KEYFRAME_EDGE},iw)':-2", str(out)]
            try:
                proc = _run(cmd, FRAME_TIMEOUT)
            except subprocess.TimeoutExpired:
                log.warning("取关键帧超时: %s @%.1fs", rec.path, at)
                continue
            if proc.returncode == 0 and out.exists() and out.stat().st_size > 0:
                frames.append(out)
            else:
                log.warning("取关键帧失败: %s @%.1fs", rec.path, at)
        return frames

    def _set_preview(self, ctx: Context, rec: FileRecord, frame: Path) -> None:
        if rec.preview_path and Path(rec.preview_path).exists():
            return
        sha = rec.sha256 or str(rec.id)
        target = ctx.config.paths.previews / sha[:2] / f"{sha}.jpg"
        target.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(frame) as im:
            im.convert("RGB").save(target, format="JPEG", quality=90)
        ctx.state.upsert_file({"path": rec.path, "preview_path": str(target)})

    def _tag_frames(self, ctx: Context, rec: FileRecord, frames: list[Path]) -> None:
        """Describe every keyframe with the vision model; tags are the union, the caption is the first frame's."""
        cfg = ctx.config.llm
        tags: list[str] = []
        first: dict[str, Any] | None = None
        for frame in frames:
            try:
                result = ctx.llm.describe(resized_jpeg(frame, cfg.max_image_edge), cfg.language)
            except Exception:
                log.warning("视频关键帧描述失败: %s", rec.path, exc_info=True)
                continue
            if first is None and isinstance(result, dict):
                first = result
            for t in (result.get("tags") or []) if isinstance(result, dict) else []:
                text = str(t).strip()
                if text and text not in tags:
                    tags.append(text)
        if first is None:
            return
        ctx.state.add_tags(rec.id, [(t, "vlm", None) for t in tags[:12]])
        scene = str(first.get("scene", "other"))
        ctx.state.set_description(rec.id, caption=str(first.get("caption", "")).strip()[:80], scene=scene,
                                  style_preset="default", model=cfg.vision_model)

    # ------------------------------------------------------------------ proxy
    def _proxy(self, ctx: Context, rec: FileRecord) -> Path:
        cfg = ctx.config
        rel = Path(rec.rel_path)
        out = cfg.paths.output / "proxies" / rel.parent / f"{rel.stem}_proxy.mp4"
        if out.exists() and out.stat().st_size > 0:
            return out
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_name(f".{out.stem}.tmp.mp4")
        encoder = self.pick_encoder(ctx)
        proc = self._encode(rec.path, tmp, encoder, cfg.video.proxy_height)
        if proc.returncode != 0 and encoder != "libx264":
            log.warning("编码器 %s 运行失败，回退 libx264: %s", encoder, (proc.stderr or "").strip()[-200:])
            self._encoders = (self._encoders or set()) - {encoder}
            proc = self._encode(rec.path, tmp, "libx264", cfg.video.proxy_height)
        if proc.returncode != 0 or not tmp.exists():
            tmp.unlink(missing_ok=True)
            raise RuntimeError(f"ffmpeg 转码失败 (返回码 {proc.returncode}): {(proc.stderr or '').strip()[-300:]}")
        os.replace(tmp, out)
        ctx.state.add_output(
            "proxy", str(out), file_id=rec.id, params={"encoder": encoder, "height": cfg.video.proxy_height}
        )
        return out

    @staticmethod
    def _encode(src: str, dst: Path, encoder: str, height: int) -> subprocess.CompletedProcess[str]:
        quality = ["-crf", "23"] if encoder == "libx264" else ["-preset", "p4", "-cq", "28"]
        cmd = ["ffmpeg", "-y", "-v", "error", "-i", src, "-vf", f"scale=-2:'min({height},ih)'", "-c:v", encoder,
               *quality, "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", str(dst)]
        return _run(cmd, ENCODE_TIMEOUT)

    # ------------------------------------------------------------------ timelapse groups (global part)
    def _timelapses(self, ctx: Context) -> None:
        state, cfg = ctx.state, ctx.config
        if ctx.dry_run:
            return
        event_of: dict[int, str] = {}
        for g in state.groups("event"):
            for fid in g["members"]:
                event_of[fid] = g.get("name") or SCATTERED
        for group in state.groups("timelapse"):
            if any(Path(o["path"]).exists() for o in state.outputs(group_id=group["id"], kind="timelapse")):
                continue
            recs = [r for r in (state.get_file(fid) for fid in group["members"]) if r is not None]
            if len(recs) < 2:
                continue
            event = sanitize_name(event_of.get(recs[0].id, SCATTERED), 40) or SCATTERED
            out = cfg.paths.output / "timelapse" / event / f"timelapse_{Path(recs[0].rel_path).stem}.mp4"
            try:
                out.parent.mkdir(parents=True, exist_ok=True)
                ctx.lumen.timelapse([r.path for r in recs], str(out), "h264")
                if not out.exists():
                    raise FileNotFoundError(f"LUMEN 没有写出延时视频: {out}")
                state.add_output("timelapse", str(out), group_id=group["id"], params={"frames": len(recs)})
            except Exception:
                log.exception("延时视频生成失败: group %s", group["id"])


STAGE = Video
