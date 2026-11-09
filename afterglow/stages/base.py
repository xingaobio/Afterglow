"""Stage framework contract.  See ARCHITECTURE.md §4.

Every pipeline stage subclasses `Stage`.  The pipeline calls `select()` to get a batch of files
whose step for this stage is still pending (and whose prerequisite stages are done), then `run()`
and records the returned `StepOutcome`s.  Global stages (`per_file = False`) are run exactly once
per pipeline run with an empty batch and do their own bookkeeping through `ctx.state`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, Literal

if TYPE_CHECKING:
    from afterglow.context import Context

STAGE_ORDER: tuple[str, ...] = (
    "backup", "index", "cull", "group", "understand", "process", "merge", "video", "publish",
)

Status = Literal["done", "failed", "skipped"]


@dataclass
class FileRecord:
    """One row of the `files` table (ARCHITECTURE.md §3).  `analysis` is joined on demand."""

    id: int
    path: str
    rel_path: str
    kind: str
    ext: str
    size: int | None = None
    mtime: float | None = None
    sha256: str | None = None
    dhash: str | None = None
    captured_at: str | None = None
    captured_src: str | None = None
    camera: str | None = None
    lens: str | None = None
    iso: int | None = None
    aperture: float | None = None
    shutter: float | None = None
    focal: float | None = None
    exposure_bias: float | None = None
    focus_distance: float | None = None
    gps_lat: float | None = None
    gps_lon: float | None = None
    width: int | None = None
    height: int | None = None
    orientation: int | None = None
    duration: float | None = None
    pair_id: int | None = None
    preview_path: str | None = None
    missing: bool = False
    analysis: dict[str, Any] | None = None

    @property
    def is_image(self) -> bool:
        return self.kind in ("raw", "jpeg", "heic", "png")

    @property
    def is_video(self) -> bool:
        return self.kind == "video"

    @property
    def pick(self) -> bool:
        return bool(self.analysis and self.analysis.get("pick"))


@dataclass
class StepOutcome:
    file_id: int
    status: Status
    error: str | None = None
    detail: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def done(cls, file_id: int, **detail: Any) -> StepOutcome:
        return cls(file_id, "done", None, detail)

    @classmethod
    def skipped(cls, file_id: int, reason: str = "") -> StepOutcome:
        return cls(file_id, "skipped", reason or None)

    @classmethod
    def failed(cls, file_id: int, error: str | BaseException) -> StepOutcome:
        text = f"{type(error).__name__}: {error}" if isinstance(error, BaseException) else str(error)
        return cls(file_id, "failed", text[:2000])


class Stage(ABC):
    """A pipeline stage.  Subclasses set the class attributes and implement `run`."""

    name: ClassVar[str]
    requires: ClassVar[tuple[str, ...]] = ()
    uses_gpu: ClassVar[bool] = False
    per_file: ClassVar[bool] = True

    def select(self, ctx: Context, limit: int) -> list[FileRecord]:
        """Files pending for this stage whose prerequisite stages are all done.

        The default asks `ctx.state.pending(self.name, limit)`; subclasses add filters (kind, pick…)
        and return `StepOutcome.skipped` from `run` for files they do not handle.
        """
        return ctx.state.pending(self.name, limit)

    @abstractmethod
    def run(self, ctx: Context, batch: list[FileRecord]) -> list[StepOutcome]:
        """Process `batch` (empty for global stages).  Never raise for a single file's failure."""

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Stage {self.name}>"


def order_stages(names: list[str] | tuple[str, ...] | None) -> list[str]:
    """Return the requested stage names in canonical order; None -> all stages."""
    if not names:
        return list(STAGE_ORDER)
    unknown = [n for n in names if n not in STAGE_ORDER]
    if unknown:
        raise ValueError(f"未知阶段: {', '.join(unknown)}；可选: {', '.join(STAGE_ORDER)}")
    return [n for n in STAGE_ORDER if n in names]
