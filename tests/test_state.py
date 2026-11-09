from __future__ import annotations

from pathlib import Path

import pytest

from afterglow.stages.base import STAGE_ORDER, StepOutcome
from afterglow.state import State


@pytest.fixture
def state(tmp_path: Path) -> State:
    return State(tmp_path / "db" / "afterglow.db")


def add(state: State, name: str, **extra: object) -> int:
    rec = {"path": f"/src/{name}", "rel_path": name, "kind": "raw", "ext": "cr3", **extra}
    return state.upsert_file(rec)


def test_schema_and_pragmas(state: State) -> None:
    assert state.get("schema_version") == "1"
    assert state._conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    assert state._conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    State(state.db_path)  # reopening is idempotent


def test_upsert_creates_pending_steps_except_backup(state: State) -> None:
    fid = add(state, "a.cr3", size=10)
    rows = state._conn.execute("SELECT stage, status FROM steps WHERE file_id=?", (fid,)).fetchall()
    assert {r["stage"] for r in rows} == set(STAGE_ORDER) - {"backup", "people"}
    assert {r["status"] for r in rows} == {"pending"}
    # second upsert updates, keeps id and step states
    state.set_step(fid, "index", "done")
    assert add(state, "a.cr3", size=20) == fid
    rec = state.get_file(fid)
    assert rec is not None and rec.size == 20 and rec.missing is False
    assert state.pending("index", 10) == []


def test_upsert_validation(state: State) -> None:
    with pytest.raises(KeyError):
        state.upsert_file({"path": "/x", "rel_path": "x", "kind": "raw", "ext": "x", "bogus": 1})
    with pytest.raises(KeyError):
        state.upsert_file({"path": "/x"})


def test_lookup_and_filters(state: State) -> None:
    a = add(state, "a.cr3")
    b = state.upsert_file({"path": "/src/b.mp4", "rel_path": "b.mp4", "kind": "video", "ext": "mp4"})
    state.set_analysis(a, pick=1, quality=80.0)
    assert state.file_by_path("/src/a.cr3").id == a  # type: ignore[union-attr]
    assert state.get_file(999) is None
    assert [r.id for r in state.files_where(kind="video")] == [b]
    assert [r.id for r in state.files_where(pick=True)] == [a]
    assert [r.id for r in state.files_where(kind=["raw", "video"])] == [a, b]
    assert state.files_where(kind=[]) == []
    assert state.get_file(a).analysis["quality"] == 80.0  # type: ignore[union-attr]
    with pytest.raises(KeyError):
        state.files_where(nope=1)


def test_mark_missing(state: State) -> None:
    a, b = add(state, "a"), add(state, "b")
    assert state.mark_missing({"/src/a"}) == 1
    assert state.get_file(b).missing is True  # type: ignore[union-attr]
    assert [r.id for r in state.pending("index", 10)] == [a]
    assert state.mark_missing({"/src/a", "/src/b"}) == 0
    assert state.get_file(b).missing is False  # type: ignore[union-attr]


def test_pending_honours_requires_and_scope(state: State) -> None:
    a, b = add(state, "a"), add(state, "b")
    assert state.pending("cull", 10, requires=("index",)) == []
    state.set_step(a, "index", "done")
    state.set_step(b, "index", "failed", "boom")
    assert [r.id for r in state.pending("cull", 10, requires=("index",))] == [a]
    state.set_step(b, "index", "skipped")
    assert [r.id for r in state.pending("cull", 10, requires=("index",))] == [a, b]
    assert [r.id for r in state.pending("cull", 1, requires=("index",))] == [a]
    assert [r.id for r in state.pending("cull", 10, file_ids=[b], requires=("index",))] == [b]
    state.set_requires_map({"cull": ("index",)})
    state.set_step(a, "index", "pending")
    assert [r.id for r in state.pending("cull", 10)] == [b]
    state.set_scope([a])
    assert state.pending("cull", 10) == []
    state.set_scope(None)
    assert [r.id for r in state.pending("cull", 10)] == [b]


def test_set_steps_attempts_and_reset(state: State) -> None:
    a, b = add(state, "a"), add(state, "b")
    state.set_steps([StepOutcome.failed(a, "x"), StepOutcome.done(b)], "index")
    state.set_steps([StepOutcome.failed(a, "y")], "index")
    row = state._conn.execute("SELECT * FROM steps WHERE file_id=? AND stage='index'", (a,)).fetchone()
    assert row["status"] == "failed" and row["attempts"] == 2 and row["error"] == "y"
    assert state.reset_stage("index") == 1
    assert [r.id for r in state.pending("index", 10)] == [a]
    assert state.reset_stage("index", only_failed=False) == 2
    with pytest.raises(ValueError):
        state.set_steps([StepOutcome(a, "weird")], "index")  # type: ignore[arg-type]


def test_analysis_groups_tags_descriptions_outputs(state: State) -> None:
    a = add(state, "a", captured_at="2024-05-01T10:00:00")
    b = add(state, "b", captured_at="2024-05-01T09:00:00")
    state.set_analysis(a, sharpness=50.0)
    state.set_analysis(a, pick=1)
    assert state.analysis(a)["sharpness"] == 50.0 and state.analysis(a)["pick"] == 1  # type: ignore[index]
    assert state.analysis(b) is None
    with pytest.raises(KeyError):
        state.set_analysis(a, nope=1)

    g = state.create_group("event", [a, b], meta={"count": 2})
    groups = state.groups("event")
    assert groups[0]["id"] == g and groups[0]["members"] == [a, b]
    assert groups[0]["started_at"] == "2024-05-01T09:00:00" and groups[0]["meta"] == {"count": 2}
    assert len(state.groups("event", unnamed_only=True)) == 1
    state.set_group_name(g, "徒步")
    assert state.groups("event", unnamed_only=True) == []

    state.add_tags(a, [("sky", "vlm", 0.9), ("sky", "rule", None)])
    state.add_tags(a, [("sky", "vlm", 0.5)])
    assert state.tags(a) == [("sky", "rule", None), ("sky", "vlm", 0.5)]

    state.set_description(a, caption="天空", scene="landscape")
    state.set_description(a, style_preset="landscape")
    d = state.description(a)
    assert d["caption"] == "天空" and d["style_preset"] == "landscape" and d["created_at"]  # type: ignore[index]

    oid = state.add_output("export", "/out/a.jpg", file_id=a, params={"q": 92})
    assert state.outputs(file_id=a)[0]["id"] == oid and state.outputs(kind="hdr") == []
    assert state.outputs(file_id=a)[0]["params"] == {"q": 92}


def test_undo_runs_kv_and_stats(state: State) -> None:
    r1 = state.start_run(["index"])
    r2 = state.start_run(["cull"])
    i1 = state.log_undo(r1, "mkdir", dst="/d")
    i2 = state.log_undo(r2, "symlink", src="/s", dst="/l", meta={"k": 1})
    entries = state.undo_entries()
    assert [e["id"] for e in entries] == [i2, i1] and entries[0]["meta"] == {"k": 1}
    assert [e["id"] for e in state.undo_entries(run_id=r1)] == [i1]
    assert state.latest_run_id(with_undo=True) == r2
    state.mark_undone([i2])
    assert [e["id"] for e in state.undo_entries()] == [i1]
    assert len(state.undo_entries(include_undone=True)) == 2
    assert state.latest_run_id(with_undo=True) == r1

    state.finish_run(r1, 3, 1, 2, "note")
    state.set("k", "v")
    state.set("k", "v2")
    assert state.get("k") == "v2" and state.get("zz", "d") == "d"

    fid = add(state, "a")
    state.set_analysis(fid, pick=1)
    state.create_group("burst", [fid])
    st = state.stats()
    assert st["files"] == {"raw": 1} and st["picks"] == 1 and st["groups"] == {"burst": 1}
    assert st["steps"]["index"]["pending"] == 1 and st["steps"]["backup"]["pending"] == 0
    assert st["runs"][0]["id"] == r2 and st["runs"][1]["ok"] == 3
