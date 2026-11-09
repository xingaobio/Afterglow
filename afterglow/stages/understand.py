"""Stage 5 - vision-model tags / captions and event naming (ARCHITECTURE.md §5 ⑤)."""

from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from PIL import Image

from afterglow.stages.base import FileRecord, Stage, StepOutcome
from afterglow.util.images import jpeg_bytes

if TYPE_CHECKING:
    from afterglow.context import Context

log = logging.getLogger(__name__)

#: Name of events without a single pick to describe.
UNSORTED = "未分类"

SCENES = {"landscape", "portrait", "street", "night", "food", "document", "screenshot", "pet", "event", "other"}
STYLES = {"landscape", "portrait", "night", "default"}
MAX_TAGS = 8
MAX_CAPTION = 80
EVENT_CAPTIONS = 6
EVENT_NAME_MAX = 12
FORBIDDEN_NAME_CHARS = re.compile(r'[/\\:*?"<>|]')


def sanitize_name(name: str, limit: int = EVENT_NAME_MAX) -> str:
    """Drop characters that are illegal in directory names and trim to `limit` characters."""
    cleaned = FORBIDDEN_NAME_CHARS.sub("", name or "").strip().strip(".")
    return re.sub(r"\s+", " ", cleaned)[:limit].strip()


def resized_jpeg(preview: str | Path, max_edge: int) -> bytes:
    """Read the stored preview and return JPEG bytes whose long edge is at most `max_edge`."""
    with Image.open(preview) as im:
        img = im.convert("RGB")
    if max(img.size) > max_edge:
        img.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
    return jpeg_bytes(img, 85)


def _place_name(meta: dict[str, Any]) -> str | None:
    center = meta.get("gps_center")
    if not center:
        return None
    try:
        import reverse_geocoder as rg  # optional extra "geo"

        hit = rg.search([(float(center[0]), float(center[1]))], mode=1)[0]
    except Exception:  # noqa: BLE001 - package missing or data file unavailable: no place name
        return None
    parts = [hit.get("name"), hit.get("admin1"), hit.get("cc")]
    return ", ".join(p for p in parts if p) or None


class Understand(Stage):
    name = "understand"
    requires = ("cull", "group")
    uses_gpu = True
    per_file = True

    def select(self, ctx: Context, limit: int) -> list[FileRecord]:
        batch = ctx.state.pending(self.name, limit, requires=self.requires)
        if not batch:
            self._name_events(ctx, final=True)
        return batch

    def run(self, ctx: Context, batch: list[FileRecord]) -> list[StepOutcome]:
        cfg = ctx.config.llm
        outcomes: dict[int, StepOutcome] = {}
        todo: list[FileRecord] = []
        for rec in batch:
            analysis = rec.analysis or ctx.state.analysis(rec.id) or {}
            if not rec.is_image:
                outcomes[rec.id] = StepOutcome.skipped(rec.id, "not an image")
            elif rec.pair_id is not None:
                outcomes[rec.id] = StepOutcome.skipped(rec.id, "paired with RAW")
            elif cfg.tag_scope == "picks" and not analysis.get("pick"):
                outcomes[rec.id] = StepOutcome.skipped(rec.id, "not a pick")
            elif not rec.preview_path or not Path(rec.preview_path).exists():
                outcomes[rec.id] = StepOutcome.failed(rec.id, "no preview image")
            else:
                todo.append(rec)

        if todo and not self._service_up(ctx):
            # One clear message instead of a traceback per photo; the files stay failed so that
            # `afterglow retry --stage understand` picks them up once the server is back.
            log.error("视觉模型服务不可用（%s），本批 %d 张照片标记为失败", cfg.server, len(todo))
            for rec in todo:
                outcomes[rec.id] = StepOutcome.failed(rec.id, f"LLM 服务不可用: {cfg.server}")
            todo = []

        with ThreadPoolExecutor(max(1, cfg.concurrency)) as pool:
            futures = [(rec, pool.submit(self._describe, ctx, rec)) for rec in todo]
            for rec, fut in futures:
                try:
                    self._store(ctx, rec, fut.result())
                    outcomes[rec.id] = StepOutcome.done(rec.id)
                except Exception as exc:
                    log.warning("视觉模型打标签失败: %s (%s: %s)", rec.path, type(exc).__name__, exc)
                    log.debug("traceback for %s", rec.path, exc_info=True)
                    outcomes[rec.id] = StepOutcome.failed(rec.id, exc)

        more = ctx.state.pending(self.name, len(batch) + 1, requires=self.requires)
        remaining = [r for r in more if r.id not in outcomes]
        self._name_events(ctx, final=not remaining)
        return [outcomes[r.id] for r in batch]

    @staticmethod
    def _service_up(ctx: Context) -> bool:
        healthy = getattr(ctx.llm, "healthy", None)
        if not callable(healthy):
            return True
        try:
            return bool(healthy())
        except Exception:  # a probe must never take the stage down
            return False

    def _describe(self, ctx: Context, rec: FileRecord) -> dict[str, Any]:
        cfg = ctx.config.llm
        data = resized_jpeg(rec.preview_path or "", cfg.max_image_edge)
        last: Exception | None = None
        for _ in range(2):  # one retry on a malformed answer
            try:
                result = ctx.llm.describe(data, cfg.language)
                if isinstance(result, dict) and str(result.get("caption", "")).strip():
                    return result
                last = ValueError(f"模型返回缺少 caption: {result!r}"[:300])
            except Exception as exc:
                last = exc
        assert last is not None
        raise last

    def _store(self, ctx: Context, rec: FileRecord, result: dict[str, Any]) -> None:
        scene = str(result.get("scene", "other")).strip().lower()
        style = str(result.get("style", "default")).strip().lower()
        raw_tags = result.get("tags") or []
        tags: list[str] = []
        for t in raw_tags if isinstance(raw_tags, list) else []:
            text = str(t).strip()
            if text and text not in tags:
                tags.append(text)
        ctx.state.add_tags(rec.id, [(t, "vlm", None) for t in tags[:MAX_TAGS]])
        ctx.state.set_description(
            rec.id,
            caption=str(result.get("caption", "")).strip()[:MAX_CAPTION],
            scene=scene if scene in SCENES else "other",
            style_preset=style if style in STYLES else "default",
            model=ctx.config.llm.vision_model,
            created_at=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        )

    # ------------------------------------------------------------------ event naming
    def _name_events(self, ctx: Context, final: bool) -> int:
        """Name events lacking a name once their picks are described (or when no files remain pending)."""
        state = ctx.state
        llm = getattr(ctx, "brain_llm", None) or ctx.llm
        language = ctx.config.llm.language
        named = 0
        for group in state.groups("event", unnamed_only=True):
            recs = [r for r in (state.get_file(m) for m in group["members"]) if r is not None]
            picks = [r for r in recs if r.is_image and (r.analysis or state.analysis(r.id) or {}).get("pick")]
            pool = picks or [r for r in recs if r.is_image]
            described = [(r, state.description(r.id)) for r in pool]
            if picks and not final and any(d is None for _, d in described):
                continue
            captions = [d["caption"] for _, d in described if d and d.get("caption")][:EVENT_CAPTIONS]
            if not captions:
                if final and not picks:
                    # Nothing in this event will ever be described (no picks): give it a plain,
                    # permanent name so `publish` can place its files.
                    state.set_group_name(group["id"], sanitize_name(_place_name(group.get("meta") or {}) or UNSORTED))
                    named += 1
                continue
            date = (group.get("started_at") or (recs[0].captured_at if recs else "") or "")[:10]
            try:
                name = sanitize_name(llm.name_event(captions, date, _place_name(group.get("meta") or {}), language))
            except Exception:
                log.exception("事件命名失败: group %s", group["id"])
                continue
            if name:
                state.set_group_name(group["id"], name)
                named += 1
        return named


STAGE = Understand
