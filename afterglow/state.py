"""SQLite state database (ARCHITECTURE.md §3).  All SQL in the project lives in this file."""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Iterable, Sequence
from dataclasses import fields
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from afterglow.stages.base import SOFT_PREREQUISITES, STAGE_ORDER, FileRecord, StepOutcome

SCHEMA_VERSION = 1

_DDL = """
CREATE TABLE IF NOT EXISTS files (
  id INTEGER PRIMARY KEY,
  path TEXT UNIQUE NOT NULL,
  rel_path TEXT NOT NULL,
  size INTEGER, mtime REAL,
  sha256 TEXT,
  dhash TEXT,
  kind TEXT NOT NULL,
  ext TEXT NOT NULL,
  captured_at TEXT,
  captured_src TEXT,
  camera TEXT, lens TEXT, iso INTEGER, aperture REAL, shutter REAL, focal REAL,
  exposure_bias REAL, focus_distance REAL,
  gps_lat REAL, gps_lon REAL,
  width INTEGER, height INTEGER, orientation INTEGER,
  duration REAL,
  pair_id INTEGER REFERENCES files(id),
  preview_path TEXT,
  missing INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS ix_files_captured ON files(captured_at);
CREATE INDEX IF NOT EXISTS ix_files_sha ON files(sha256);

CREATE TABLE IF NOT EXISTS steps (
  file_id INTEGER NOT NULL REFERENCES files(id),
  stage TEXT NOT NULL,
  status TEXT NOT NULL,
  attempts INTEGER DEFAULT 0,
  error TEXT,
  updated_at TEXT NOT NULL,
  PRIMARY KEY (file_id, stage)
);
CREATE INDEX IF NOT EXISTS ix_steps_stage_status ON steps(stage, status);

CREATE TABLE IF NOT EXISTS groups (
  id INTEGER PRIMARY KEY,
  kind TEXT NOT NULL,
  name TEXT,
  started_at TEXT, ended_at TEXT,
  meta TEXT
);

CREATE TABLE IF NOT EXISTS analysis (
  file_id INTEGER PRIMARY KEY REFERENCES files(id),
  sharpness REAL, faces INTEGER, eyes_closed INTEGER, blurry INTEGER,
  exposure_score REAL,
  quality REAL,
  burst_id INTEGER REFERENCES groups(id),
  burst_best INTEGER DEFAULT 0,
  pick INTEGER DEFAULT 0,
  reject_reason TEXT
);

CREATE TABLE IF NOT EXISTS group_members (
  group_id INTEGER NOT NULL REFERENCES groups(id),
  file_id INTEGER NOT NULL REFERENCES files(id),
  position INTEGER NOT NULL,
  PRIMARY KEY (group_id, file_id)
);

CREATE TABLE IF NOT EXISTS tags (
  file_id INTEGER NOT NULL REFERENCES files(id),
  tag TEXT NOT NULL,
  source TEXT NOT NULL,
  confidence REAL,
  PRIMARY KEY (file_id, tag, source)
);

CREATE TABLE IF NOT EXISTS descriptions (
  file_id INTEGER PRIMARY KEY REFERENCES files(id),
  caption TEXT, scene TEXT,
  style_preset TEXT,
  model TEXT, created_at TEXT
);

CREATE TABLE IF NOT EXISTS outputs (
  id INTEGER PRIMARY KEY,
  file_id INTEGER REFERENCES files(id),
  group_id INTEGER REFERENCES groups(id),
  kind TEXT NOT NULL,
  path TEXT NOT NULL,
  params TEXT,
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS runs (
  id INTEGER PRIMARY KEY,
  started_at TEXT NOT NULL, finished_at TEXT,
  stages TEXT NOT NULL,
  ok INTEGER DEFAULT 0, failed INTEGER DEFAULT 0, skipped INTEGER DEFAULT 0,
  note TEXT
);

CREATE TABLE IF NOT EXISTS undo_log (
  id INTEGER PRIMARY KEY,
  run_id INTEGER REFERENCES runs(id),
  ts TEXT NOT NULL,
  action TEXT NOT NULL,
  src TEXT, dst TEXT,
  meta TEXT,
  undone INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS kv (key TEXT PRIMARY KEY, value TEXT);
"""

_FILE_COLUMNS = tuple(f.name for f in fields(FileRecord) if f.name != "analysis")
_ANALYSIS_COLUMNS = (
    "sharpness", "faces", "eyes_closed", "blurry", "exposure_score", "quality",
    "burst_id", "burst_best", "pick", "reject_reason",
)
_DESCRIPTION_COLUMNS = ("caption", "scene", "style_preset", "model", "created_at")
_STEP_STATUSES = ("pending", "done", "failed", "skipped")


def now_iso() -> str:
    """Current UTC time as an ISO-8601 string."""
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _json_or_none(value: dict[str, Any] | None) -> str | None:
    return None if value is None else json.dumps(value, ensure_ascii=False, sort_keys=True)


def _loads(text: str | None) -> Any:
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


class State:
    """Thread-safe wrapper around the Afterglow SQLite database."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False, timeout=60.0)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._scope: set[int] | None = None
        self._requires: dict[str, tuple[str, ...]] = {}
        self._migrate()

    # ------------------------------------------------------------------ plumbing
    def _migrate(self) -> None:
        with self._lock:
            self._conn.executescript(_DDL)
            row = self._conn.execute("SELECT value FROM kv WHERE key='schema_version'").fetchone()
            if row is None:
                self._conn.execute("INSERT INTO kv(key, value) VALUES('schema_version', ?)", (str(SCHEMA_VERSION),))
            elif int(row["value"]) > SCHEMA_VERSION:
                raise RuntimeError(f"数据库版本 {row['value']} 比本程序支持的 {SCHEMA_VERSION} 更新")
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def _execute(self, sql: str, params: Sequence[Any] = ()) -> sqlite3.Cursor:
        """Run one write statement and commit."""
        with self._lock:
            try:
                cur = self._conn.execute(sql, params)
                self._conn.commit()
            except BaseException:
                self._conn.rollback()
                raise
            return cur

    def _query(self, sql: str, params: Sequence[Any] = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(sql, params).fetchall()

    def set_scope(self, file_ids: Iterable[int] | None) -> None:
        """Restrict `pending()` to these file ids (used by `run --only`); None clears the restriction."""
        self._scope = None if file_ids is None else set(file_ids)

    def set_requires_map(self, mapping: dict[str, tuple[str, ...]]) -> None:
        """Default prerequisites per stage, used by `pending()` when `requires` is not passed."""
        self._requires = {k: tuple(v) for k, v in mapping.items()}

    # ------------------------------------------------------------------ files
    def _records(self, rows: Sequence[sqlite3.Row]) -> list[FileRecord]:
        if not rows:
            return []
        ids = [r["id"] for r in rows]
        analyses: dict[int, dict[str, Any]] = {}
        for i in range(0, len(ids), 500):
            chunk = ids[i : i + 500]
            marks = ",".join("?" * len(chunk))
            for a in self._query(f"SELECT * FROM analysis WHERE file_id IN ({marks})", chunk):
                d = dict(a)
                analyses[d.pop("file_id")] = d
        out: list[FileRecord] = []
        for r in rows:
            data = {k: r[k] for k in _FILE_COLUMNS}
            data["missing"] = bool(data["missing"])
            out.append(FileRecord(**data, analysis=analyses.get(r["id"])))
        return out

    def upsert_file(self, rec: dict[str, Any]) -> int:
        """Insert or update a file by `path`; a new file gets pending steps for all stages but backup."""
        unknown = set(rec) - set(_FILE_COLUMNS)
        if unknown:
            raise KeyError(f"未知的 files 字段: {sorted(unknown)}")
        if "path" not in rec:
            raise KeyError("upsert_file 需要 path")
        values = {k: v for k, v in rec.items() if k != "id"}
        if "missing" in values:
            values["missing"] = int(bool(values["missing"]))
        with self._lock:
            try:
                row = self._conn.execute("SELECT id FROM files WHERE path=?", (values["path"],)).fetchone()
                if row is not None:
                    file_id = int(row["id"])
                    sets = {k: v for k, v in values.items() if k != "path"}
                    sets.setdefault("missing", 0)
                    assignments = ",".join(f"{k}=?" for k in sets)
                    self._conn.execute(f"UPDATE files SET {assignments} WHERE id=?", [*sets.values(), file_id])
                else:
                    for needed in ("rel_path", "kind", "ext"):
                        if needed not in values:
                            raise KeyError(f"新文件需要 {needed}")
                    cols = ",".join(values)
                    marks = ",".join("?" * len(values))
                    cur = self._conn.execute(f"INSERT INTO files({cols}) VALUES({marks})", list(values.values()))
                    file_id = int(cur.lastrowid or 0)
                    ts = now_iso()
                    self._conn.executemany(
                        "INSERT INTO steps(file_id, stage, status, attempts, updated_at) VALUES(?,?,'pending',0,?)",
                        [(file_id, s, ts) for s in STAGE_ORDER if s != "backup"],
                    )
                self._conn.commit()
            except BaseException:
                self._conn.rollback()
                raise
            return file_id

    def get_file(self, file_id: int) -> FileRecord | None:
        recs = self._records(self._query("SELECT * FROM files WHERE id=?", (file_id,)))
        return recs[0] if recs else None

    def file_by_path(self, path: str) -> FileRecord | None:
        recs = self._records(self._query("SELECT * FROM files WHERE path=?", (str(path),)))
        return recs[0] if recs else None

    def files_where(self, **filters: Any) -> list[FileRecord]:
        """Files matching column equality filters (`kind=`, `missing=`, `pick=`; list -> IN; None -> IS NULL)."""
        clauses: list[str] = []
        params: list[Any] = []
        join = ""
        for key, value in filters.items():
            if key == "pick":
                join = "LEFT JOIN analysis a ON a.file_id = f.id"
                col = "COALESCE(a.pick, 0)"
                value = int(bool(value))
            elif key in _FILE_COLUMNS:
                col = f"f.{key}"
                if key == "missing":
                    value = int(bool(value))
            else:
                raise KeyError(f"未知的过滤字段: {key}")
            if value is None:
                clauses.append(f"{col} IS NULL")
            elif isinstance(value, (list, tuple, set)):
                items = list(value)
                clauses.append(f"{col} IN ({','.join('?' * len(items))})" if items else "0")
                params.extend(items)
            else:
                clauses.append(f"{col} = ?")
                params.append(value)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self._query(f"SELECT f.* FROM files f {join} {where} ORDER BY f.id", params)
        return self._records(rows)

    def mark_missing(self, present_paths: set[str]) -> int:
        """Flag files not in `present_paths` as missing (and clear the flag for those present).

        Returns the number of files newly flagged missing.
        """
        newly = 0
        with self._lock:
            try:
                for row in self._conn.execute("SELECT id, path, missing FROM files").fetchall():
                    here = row["path"] in present_paths
                    if not here and not row["missing"]:
                        self._conn.execute("UPDATE files SET missing=1 WHERE id=?", (row["id"],))
                        newly += 1
                    elif here and row["missing"]:
                        self._conn.execute("UPDATE files SET missing=0 WHERE id=?", (row["id"],))
                self._conn.commit()
            except BaseException:
                self._conn.rollback()
                raise
        return newly

    # ------------------------------------------------------------------ steps
    def pending(
        self,
        stage: str,
        limit: int,
        file_ids: list[int] | None = None,
        requires: Sequence[str] = (),
    ) -> list[FileRecord]:
        """Files whose `stage` step is pending and whose prerequisites are done/skipped."""
        req = [r for r in (tuple(requires) or self._requires.get(stage, ())) if r != "backup"]
        sql = (
            "SELECT f.* FROM files f JOIN steps s ON s.file_id=f.id AND s.stage=? AND s.status='pending' "
            "WHERE f.missing=0"
        )
        params: list[Any] = [stage]
        for r in req:
            # Soft prerequisites (ARCHITECTURE.md §4: a failed `understand` must not block processing)
            # only need to have been attempted; hard ones must be done or skipped.
            statuses = "('done','skipped','failed')" if r in SOFT_PREREQUISITES else "('done','skipped')"
            sql += (
                " AND EXISTS (SELECT 1 FROM steps p WHERE p.file_id=f.id AND p.stage=? "
                f"AND p.status IN {statuses})"
            )
            params.append(r)
        ids = file_ids
        if ids is None and self._scope is not None:
            ids = sorted(self._scope)
        if ids is not None:
            if not ids:
                return []
            sql += f" AND f.id IN ({','.join('?' * len(ids))})"
            params.extend(ids)
        sql += " ORDER BY f.id LIMIT ?"
        params.append(int(limit))
        return self._records(self._query(sql, params))

    _STEP_UPSERT = (
        "INSERT INTO steps(file_id, stage, status, attempts, error, updated_at) VALUES(?,?,?,?,?,?) "
        "ON CONFLICT(file_id, stage) DO UPDATE SET status=excluded.status, error=excluded.error, "
        "updated_at=excluded.updated_at, attempts=steps.attempts + excluded.attempts"
    )

    def set_step(self, file_id: int, stage: str, status: str, error: str | None = None) -> None:
        self.set_steps([StepOutcome(file_id, status, error)], stage)  # type: ignore[arg-type]

    def set_steps(self, outcomes: list[StepOutcome], stage: str) -> None:
        """Record outcomes for one stage in a single transaction; failures bump `attempts`."""
        for o in outcomes:
            if o.status not in _STEP_STATUSES:
                raise ValueError(f"无效的步骤状态: {o.status}")
        if not outcomes:
            return
        ts = now_iso()
        with self._lock:
            try:
                self._conn.executemany(
                    self._STEP_UPSERT,
                    [(o.file_id, stage, o.status, 1 if o.status == "failed" else 0, o.error, ts) for o in outcomes],
                )
                self._conn.commit()
            except BaseException:
                self._conn.rollback()
                raise

    def reset_stage(self, stage: str, only_failed: bool = True) -> int:
        sql = "UPDATE steps SET status='pending', error=NULL, updated_at=? WHERE stage=?"
        if only_failed:
            sql += " AND status='failed'"
        return self._execute(sql, (now_iso(), stage)).rowcount

    def relocate_file(self, file_id: int, new_path: str, new_rel_path: str | None = None) -> None:
        """Record that a file was physically moved (organize.mode = move) so index does not mark it missing."""
        self._execute(
            "UPDATE files SET path=?, rel_path=COALESCE(?, rel_path), missing=0 WHERE id=?",
            (new_path, new_rel_path, file_id),
        )

    # ------------------------------------------------------------------ analysis / groups / tags / ...
    def set_analysis(self, file_id: int, **fields_: Any) -> None:
        unknown = set(fields_) - set(_ANALYSIS_COLUMNS)
        if unknown:
            raise KeyError(f"未知的 analysis 字段: {sorted(unknown)}")
        cols = ["file_id", *fields_]
        updates = ",".join(f"{c}=excluded.{c}" for c in fields_)
        sql = f"INSERT INTO analysis({','.join(cols)}) VALUES({','.join('?' * len(cols))})"
        sql += f" ON CONFLICT(file_id) DO UPDATE SET {updates}" if fields_ else " ON CONFLICT(file_id) DO NOTHING"
        self._execute(sql, [file_id, *fields_.values()])

    def analysis(self, file_id: int) -> dict[str, Any] | None:
        rows = self._query("SELECT * FROM analysis WHERE file_id=?", (file_id,))
        if not rows:
            return None
        d = dict(rows[0])
        d.pop("file_id", None)
        return d

    def create_group(
        self, kind: str, member_ids: list[int], name: str | None = None, meta: dict[str, Any] | None = None
    ) -> int:
        with self._lock:
            try:
                times: list[str] = []
                if member_ids:
                    marks = ",".join("?" * len(member_ids))
                    times = sorted(
                        r["captured_at"]
                        for r in self._conn.execute(
                            f"SELECT captured_at FROM files WHERE id IN ({marks}) AND captured_at IS NOT NULL",
                            member_ids,
                        )
                    )
                cur = self._conn.execute(
                    "INSERT INTO groups(kind, name, started_at, ended_at, meta) VALUES(?,?,?,?,?)",
                    (kind, name, times[0] if times else None, times[-1] if times else None, _json_or_none(meta)),
                )
                group_id = int(cur.lastrowid or 0)
                self._conn.executemany(
                    "INSERT OR REPLACE INTO group_members(group_id, file_id, position) VALUES(?,?,?)",
                    [(group_id, fid, pos) for pos, fid in enumerate(member_ids)],
                )
                self._conn.commit()
            except BaseException:
                self._conn.rollback()
                raise
            return group_id

    def groups(self, kind: str, unnamed_only: bool = False) -> list[dict[str, Any]]:
        sql = "SELECT * FROM groups WHERE kind=?"
        if unnamed_only:
            sql += " AND name IS NULL"
        out: list[dict[str, Any]] = []
        for row in self._query(sql + " ORDER BY id", (kind,)):
            d = dict(row)
            d["meta"] = _loads(d["meta"]) or {}
            d["members"] = [
                m["file_id"]
                for m in self._query(
                    "SELECT file_id FROM group_members WHERE group_id=? ORDER BY position", (d["id"],)
                )
            ]
            out.append(d)
        return out

    def set_group_name(self, group_id: int, name: str) -> None:
        self._execute("UPDATE groups SET name=? WHERE id=?", (name, group_id))

    def add_tags(self, file_id: int, tags: list[tuple[str, str, float | None]]) -> None:
        if not tags:
            return
        with self._lock:
            try:
                self._conn.executemany(
                    "INSERT OR REPLACE INTO tags(file_id, tag, source, confidence) VALUES(?,?,?,?)",
                    [(file_id, t, s, c) for t, s, c in tags],
                )
                self._conn.commit()
            except BaseException:
                self._conn.rollback()
                raise

    def tags(self, file_id: int) -> list[tuple[str, str, float | None]]:
        rows = self._query("SELECT tag, source, confidence FROM tags WHERE file_id=? ORDER BY tag", (file_id,))
        return [(r["tag"], r["source"], r["confidence"]) for r in rows]

    def set_description(self, file_id: int, **fields_: Any) -> None:
        unknown = set(fields_) - set(_DESCRIPTION_COLUMNS)
        if unknown:
            raise KeyError(f"未知的 descriptions 字段: {sorted(unknown)}")
        fields_.setdefault("created_at", now_iso())
        cols = ["file_id", *fields_]
        updates = ",".join(f"{c}=excluded.{c}" for c in fields_)
        sql = (
            f"INSERT INTO descriptions({','.join(cols)}) VALUES({','.join('?' * len(cols))}) "
            f"ON CONFLICT(file_id) DO UPDATE SET {updates}"
        )
        self._execute(sql, [file_id, *fields_.values()])

    def description(self, file_id: int) -> dict[str, Any] | None:
        rows = self._query("SELECT * FROM descriptions WHERE file_id=?", (file_id,))
        if not rows:
            return None
        d = dict(rows[0])
        d.pop("file_id", None)
        return d

    def add_output(
        self,
        kind: str,
        path: str,
        file_id: int | None = None,
        group_id: int | None = None,
        params: dict[str, Any] | None = None,
    ) -> int:
        cur = self._execute(
            "INSERT INTO outputs(file_id, group_id, kind, path, params, created_at) VALUES(?,?,?,?,?,?)",
            (file_id, group_id, kind, str(path), _json_or_none(params), now_iso()),
        )
        return int(cur.lastrowid or 0)

    def outputs(self, file_id: int | None = None, group_id: int | None = None, kind: str | None = None) -> list[dict]:
        clauses, params = [], []
        for col, val in (("file_id", file_id), ("group_id", group_id), ("kind", kind)):
            if val is not None:
                clauses.append(f"{col}=?")
                params.append(val)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        out = []
        for row in self._query(f"SELECT * FROM outputs {where} ORDER BY id", params):
            d = dict(row)
            d["params"] = _loads(d["params"]) or {}
            out.append(d)
        return out

    # ------------------------------------------------------------------ undo / runs / kv
    def log_undo(
        self,
        run_id: int,
        action: str,
        src: str | None = None,
        dst: str | None = None,
        meta: dict[str, Any] | None = None,
    ) -> int:
        cur = self._execute(
            "INSERT INTO undo_log(run_id, ts, action, src, dst, meta) VALUES(?,?,?,?,?,?)",
            (run_id, now_iso(), action, None if src is None else str(src), None if dst is None else str(dst),
             _json_or_none(meta)),
        )
        return int(cur.lastrowid or 0)

    def undo_entries(self, run_id: int | None = None, include_undone: bool = False) -> list[dict[str, Any]]:
        clauses, params = [], []
        if run_id is not None:
            clauses.append("run_id=?")
            params.append(run_id)
        if not include_undone:
            clauses.append("undone=0")
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        out = []
        for row in self._query(f"SELECT * FROM undo_log {where} ORDER BY id DESC", params):
            d = dict(row)
            d["meta"] = _loads(d["meta"]) or {}
            out.append(d)
        return out

    def mark_undone(self, ids: list[int]) -> None:
        if not ids:
            return
        with self._lock:
            try:
                self._conn.executemany("UPDATE undo_log SET undone=1 WHERE id=?", [(i,) for i in ids])
                self._conn.commit()
            except BaseException:
                self._conn.rollback()
                raise

    def latest_run_id(self, with_undo: bool = False) -> int | None:
        """Id of the newest run (optionally: the newest run that has un-undone undo entries)."""
        if with_undo:
            rows = self._query("SELECT MAX(run_id) AS m FROM undo_log WHERE undone=0 AND run_id IS NOT NULL")
        else:
            rows = self._query("SELECT MAX(id) AS m FROM runs")
        return rows[0]["m"] if rows and rows[0]["m"] is not None else None

    def start_run(self, stages: list[str]) -> int:
        cur = self._execute("INSERT INTO runs(started_at, stages) VALUES(?,?)", (now_iso(), ",".join(stages)))
        return int(cur.lastrowid or 0)

    def finish_run(self, run_id: int, ok: int, failed: int, skipped: int, note: str = "") -> None:
        self._execute(
            "UPDATE runs SET finished_at=?, ok=?, failed=?, skipped=?, note=? WHERE id=?",
            (now_iso(), ok, failed, skipped, note, run_id),
        )

    def get(self, key: str, default: str | None = None) -> str | None:
        rows = self._query("SELECT value FROM kv WHERE key=?", (key,))
        return rows[0]["value"] if rows else default

    def set(self, key: str, value: str) -> None:
        self._execute("INSERT INTO kv(key, value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                      (key, value))

    # ------------------------------------------------------------------ stats
    def newest_file_mtime(self) -> float | None:
        rows = self._query("SELECT MAX(mtime) AS m FROM files")
        return rows[0]["m"] if rows else None

    def stats(self) -> dict[str, Any]:
        steps: dict[str, dict[str, int]] = {s: dict.fromkeys(_STEP_STATUSES, 0) for s in STAGE_ORDER}
        for r in self._query("SELECT stage, status, COUNT(*) AS n FROM steps GROUP BY stage, status"):
            steps.setdefault(r["stage"], dict.fromkeys(_STEP_STATUSES, 0))[r["status"]] = r["n"]
        files = {r["kind"]: r["n"] for r in self._query("SELECT kind, COUNT(*) AS n FROM files GROUP BY kind")}
        groups = {r["kind"]: r["n"] for r in self._query("SELECT kind, COUNT(*) AS n FROM groups GROUP BY kind")}
        picks = self._query("SELECT COUNT(*) AS n FROM analysis WHERE pick=1")[0]["n"]
        missing = self._query("SELECT COUNT(*) AS n FROM files WHERE missing=1")[0]["n"]
        runs = [dict(r) for r in self._query("SELECT * FROM runs ORDER BY id DESC LIMIT 5")]
        return {
            "steps": steps,
            "files": files,
            "files_total": sum(files.values()),
            "missing": missing,
            "picks": picks,
            "groups": groups,
            "runs": runs,
        }
