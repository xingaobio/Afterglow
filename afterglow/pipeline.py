"""Stage registry and scheduler (ARCHITECTURE.md §4)."""

from __future__ import annotations

import contextlib
import importlib
import inspect
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from afterglow.gpu_lock import GpuLock
from afterglow.stages.base import STAGE_ORDER, FileRecord, Stage, StepOutcome, order_stages

if TYPE_CHECKING:
    from afterglow.context import Context

log = logging.getLogger(__name__)

REGISTRY: dict[str, type[Stage]] = {}
MAX_CONSECUTIVE_ERRORS = 20


@dataclass
class StageReport:
    name: str
    ok: int = 0
    failed: int = 0
    skipped: int = 0
    aborted: bool = False
    error: str | None = None

    def count(self, outcomes: list[StepOutcome]) -> None:
        for o in outcomes:
            if o.status == "done":
                self.ok += 1
            elif o.status == "failed":
                self.failed += 1
            else:
                self.skipped += 1


@dataclass
class RunReport:
    run_id: int
    stages: dict[str, StageReport] = field(default_factory=dict)
    note: str = ""

    @property
    def ok(self) -> int:
        return sum(s.ok for s in self.stages.values())

    @property
    def failed(self) -> int:
        return sum(s.failed for s in self.stages.values())

    @property
    def skipped(self) -> int:
        return sum(s.skipped for s in self.stages.values())


def load_stages() -> dict[str, type[Stage]]:
    """Import afterglow.stages.<name> for every canonical stage and register its Stage subclass.

    A module that fails to import is logged and skipped; names already registered are kept.
    """
    for name in STAGE_ORDER:
        if name in REGISTRY:
            continue
        module_name = f"afterglow.stages.{name}"
        try:
            module = importlib.import_module(module_name)
        except ImportError as exc:
            log.warning("阶段模块 %s 无法导入，已跳过: %s", name, exc)
            continue
        found = [
            obj
            for _, obj in inspect.getmembers(module, inspect.isclass)
            if issubclass(obj, Stage) and obj is not Stage and not inspect.isabstract(obj)
            and getattr(obj, "name", None) == name
        ]
        if not found:
            log.warning("module %s defines no Stage named %r", module_name, name)
            continue
        REGISTRY[name] = found[0]
    return REGISTRY


def _scope_ids(ctx: Context, only_paths: list[str] | None) -> list[int] | None:
    if only_paths is None:
        return None
    roots = [str(Path(p).expanduser().absolute()) for p in only_paths]
    ids: list[int] = []
    for rec in ctx.state.files_where():
        if any(rec.path == r or rec.path.startswith(r.rstrip("/") + "/") for r in roots):
            ids.append(rec.id)
    return ids


def _normalise(batch: list[FileRecord], outcomes: list[StepOutcome]) -> list[StepOutcome]:
    """One outcome per batch file: last wins; files without an outcome fail."""
    by_id = {o.file_id: o for o in outcomes}
    result: list[StepOutcome] = []
    for rec in batch:
        result.append(by_id.pop(rec.id, None) or StepOutcome.failed(rec.id, "stage returned no outcome"))
    result.extend(by_id.values())
    return result


def _run_per_file(ctx: Context, stage: Stage, batch_size: int, rep: StageReport) -> None:
    errors_in_row = 0
    while True:
        batch = stage.select(ctx, batch_size)
        if not batch:
            return
        try:
            outcomes = stage.run(ctx, batch)
            errors_in_row = 0
        except Exception as exc:
            errors_in_row += 1
            log.exception("stage %s raised on a batch of %d files", stage.name, len(batch))
            outcomes = [StepOutcome.failed(rec.id, exc) for rec in batch]
            if errors_in_row >= MAX_CONSECUTIVE_ERRORS:
                rep.aborted = True
                rep.error = f"连续 {MAX_CONSECUTIVE_ERRORS} 次异常，阶段中止"
        outcomes = _normalise(batch, outcomes)
        rep.count(outcomes)
        if ctx.dry_run:
            log.info("[dry-run] 阶段 %s 首批 %d 个文件已模拟，不写入状态", stage.name, len(batch))
            return
        ctx.state.set_steps(outcomes, stage.name)
        if rep.aborted:
            log.error("阶段 %s: %s", stage.name, rep.error)
            return


def _run_global(ctx: Context, stage: Stage, rep: StageReport) -> None:
    try:
        outcomes = stage.run(ctx, [])
    except Exception as exc:
        log.exception("global stage %s raised", stage.name)
        rep.failed += 1
        rep.error = f"{type(exc).__name__}: {exc}"
        return
    rep.count(outcomes)
    if outcomes and not ctx.dry_run:
        ctx.state.set_steps(outcomes, stage.name)


def run(
    ctx: Context,
    stages: list[str] | None = None,
    batch_size: int | None = None,
    only_paths: list[str] | None = None,
) -> RunReport:
    """Run the requested stages (canonical order) and record everything in the `runs` table."""
    names = order_stages(stages)
    load_stages()
    available = [n for n in names if n in REGISTRY]
    for n in names:
        if n not in REGISTRY:
            log.warning("阶段 %s 未注册，已跳过", n)
    size = batch_size or ctx.config.run.batch_size
    run_id = ctx.state.start_run(available)
    ctx.run_id = run_id
    report = RunReport(run_id=run_id)
    ctx.state.set_requires_map({n: tuple(REGISTRY[n].requires) for n in REGISTRY})
    ctx.state.set_scope(_scope_ids(ctx, only_paths))
    try:
        for name in available:
            cls = REGISTRY[name]
            stage = cls()
            rep = report.stages[name] = StageReport(name)
            log.info("开始阶段: %s", name)
            guard = GpuLock(ctx) if cls.uses_gpu else contextlib.nullcontext()
            try:
                with guard:
                    if cls.per_file:
                        _run_per_file(ctx, stage, size, rep)
                    else:
                        _run_global(ctx, stage, rep)
            except Exception as exc:
                log.exception("stage %s aborted", name)
                rep.aborted = True
                rep.error = f"{type(exc).__name__}: {exc}"
            log.info("阶段 %s 完成: 成功 %d，失败 %d，跳过 %d", name, rep.ok, rep.failed, rep.skipped)
    except BaseException:
        report.note = "interrupted"
        raise
    finally:
        aborted = [n for n, r in report.stages.items() if r.aborted]
        if aborted:
            report.note = (report.note + " aborted: " + ",".join(aborted)).strip()
        ctx.state.set_scope(None)
        ctx.state.finish_run(run_id, report.ok, report.failed, report.skipped, report.note)
    return report
