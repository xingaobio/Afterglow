from __future__ import annotations

import logging
from pathlib import Path
from typing import ClassVar

import pytest

from afterglow import config, pipeline
from afterglow.context import Context
from afterglow.stages.base import FileRecord, Stage, StepOutcome
from afterglow.state import State


def make_ctx(tmp_path: Path, n_files: int = 5) -> Context:
    (tmp_path / "source").mkdir(exist_ok=True)
    cfg = config.from_dict({
        "paths": {"source": str(tmp_path / "source"), "work": str(tmp_path / "work")},
        "gpu": {"lock_file": str(tmp_path / "gpu.lock")},
    })
    state = State(tmp_path / "work" / "afterglow.db")
    for i in range(n_files):
        state.upsert_file({"path": str(tmp_path / "source" / f"{i}.jpg"), "rel_path": f"{i}.jpg", "kind": "jpeg",
                           "ext": "jpg"})
    return Context(config=cfg, state=state, log=logging.getLogger("test"))


class FakeIndex(Stage):
    name: ClassVar[str] = "index"
    seen: ClassVar[list[int]] = []

    def run(self, ctx: Context, batch: list[FileRecord]) -> list[StepOutcome]:
        FakeIndex.seen.append(len(batch))
        out = []
        for rec in batch:
            if rec.rel_path == "1.jpg":
                out.append(StepOutcome.failed(rec.id, "bad file"))
            elif rec.rel_path == "2.jpg":
                out.append(StepOutcome.skipped(rec.id, "meh"))
            elif rec.rel_path == "3.jpg":
                continue  # no outcome at all
            else:
                out.append(StepOutcome.done(rec.id))
        return out


class FakeGroup(Stage):
    name: ClassVar[str] = "group"
    requires: ClassVar[tuple[str, ...]] = ("index",)
    per_file: ClassVar[bool] = False
    calls: ClassVar[int] = 0

    def run(self, ctx: Context, batch: list[FileRecord]) -> list[StepOutcome]:
        FakeGroup.calls += 1
        assert batch == []
        return [StepOutcome.done(r.id) for r in ctx.state.files_where()]


class FakeCull(Stage):
    name: ClassVar[str] = "cull"
    requires: ClassVar[tuple[str, ...]] = ("index",)

    def run(self, ctx: Context, batch: list[FileRecord]) -> list[StepOutcome]:
        raise RuntimeError("kaboom")


@pytest.fixture(autouse=True)
def registry(monkeypatch: pytest.MonkeyPatch) -> dict[str, type[Stage]]:
    reg: dict[str, type[Stage]] = {"index": FakeIndex, "cull": FakeCull, "group": FakeGroup}
    monkeypatch.setattr(pipeline, "REGISTRY", reg)
    monkeypatch.setattr(pipeline, "load_stages", lambda: reg)
    FakeIndex.seen = []
    FakeGroup.calls = 0
    return reg


def status_counts(ctx: Context, stage: str) -> dict[str, int]:
    return ctx.state.stats()["steps"][stage]


def test_full_run(tmp_path: Path) -> None:
    ctx = make_ctx(tmp_path)
    report = pipeline.run(ctx, batch_size=2)
    assert list(report.stages) == ["index", "cull", "group"]  # canonical order
    assert FakeIndex.seen == [2, 2, 1]
    idx = status_counts(ctx, "index")
    assert (idx["done"], idx["failed"], idx["skipped"], idx["pending"]) == (2, 2, 1, 0)
    # the file with no outcome was failed explicitly
    failed = ctx.state._conn.execute(
        "SELECT error FROM steps WHERE stage='index' AND status='failed' ORDER BY file_id").fetchall()
    assert {r["error"] for r in failed} == {"bad file", "stage returned no outcome"}
    # cull only sees files whose index is done/skipped (3 files) and fails them all
    cull = status_counts(ctx, "cull")
    assert (cull["failed"], cull["pending"]) == (3, 2)
    assert report.stages["cull"].failed == 3 and not report.stages["cull"].aborted
    # global stage ran exactly once
    assert FakeGroup.calls == 1
    assert status_counts(ctx, "group")["done"] == 5

    runs = ctx.state.stats()["runs"]
    assert len(runs) == 1 and runs[0]["finished_at"] and runs[0]["stages"] == "index,cull,group"
    assert (runs[0]["ok"], runs[0]["failed"], runs[0]["skipped"]) == (report.ok, report.failed, report.skipped)
    assert report.failed == 2 + 3 and report.ok == 2 + 5 and report.skipped == 1

    # re-running processes nothing new for index (done/failed/skipped are not pending)
    FakeIndex.seen = []
    pipeline.run(ctx, stages=["index"])
    assert FakeIndex.seen == []


def test_stage_subset_and_only_paths(tmp_path: Path) -> None:
    ctx = make_ctx(tmp_path)
    report = pipeline.run(ctx, stages=["index"], only_paths=[str(tmp_path / "source" / "0.jpg")])
    assert list(report.stages) == ["index"]
    assert status_counts(ctx, "index")["done"] == 1 and status_counts(ctx, "index")["pending"] == 4
    assert ctx.state.pending("index", 10) != []  # scope cleared after the run


def test_dry_run_records_nothing(tmp_path: Path) -> None:
    ctx = make_ctx(tmp_path)
    ctx.dry_run = True
    pipeline.run(ctx, stages=["index"])
    assert status_counts(ctx, "index")["pending"] == 5


def test_unknown_stage(tmp_path: Path) -> None:
    ctx = make_ctx(tmp_path)
    with pytest.raises(ValueError):
        pipeline.run(ctx, stages=["nope"])


def test_abort_after_20_consecutive_exceptions(tmp_path: Path) -> None:
    ctx = make_ctx(tmp_path, n_files=0)
    # batch of 1 file and the stage "forgets" to mark them: use a stage whose select ignores status
    class Sticky(FakeCull):
        name: ClassVar[str] = "cull"
        requires: ClassVar[tuple[str, ...]] = ()

        def select(self, ctx: Context, limit: int) -> list[FileRecord]:
            return ctx.state.files_where()[:limit]

    ctx.state.upsert_file({"path": str(tmp_path / "source" / "z.jpg"), "rel_path": "z.jpg", "kind": "jpeg",
                           "ext": "jpg"})
    pipeline.REGISTRY["cull"] = Sticky
    report = pipeline.run(ctx, stages=["cull"], batch_size=1)
    assert report.stages["cull"].aborted
    assert report.stages["cull"].failed == 20
    assert "aborted" in ctx.state.stats()["runs"][0]["note"]


def test_load_stages_tolerates_missing_modules(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pipeline, "REGISTRY", {})
    reg = pipeline.load_stages()  # real function; partial tree is fine
    assert isinstance(reg, dict)
    assert all(issubclass(cls, Stage) for cls in reg.values())
