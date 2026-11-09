"""Undo log: record reversible actions and replay them backwards (ARCHITECTURE.md §7)."""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from afterglow.context import Context


@dataclass
class UndoReport:
    undone: int = 0
    skipped: int = 0
    errors: list[str] = field(default_factory=list)
    planned: list[str] = field(default_factory=list)  # human-readable actions (dry run)


class UndoLog:
    def __init__(self, ctx: Context) -> None:
        self.ctx = ctx

    def record(
        self,
        action: str,
        src: str | os.PathLike[str] | None = None,
        dst: str | os.PathLike[str] | None = None,
        meta: dict[str, Any] | None = None,
    ) -> int:
        """Append an undo entry for the current run and return its id."""
        return self.ctx.state.log_undo(
            self.ctx.run_id, action, None if src is None else str(src), None if dst is None else str(dst), meta
        )

    def undo(self, run_id: int | None = None, last: bool = False, dry_run: bool = False) -> UndoReport:
        """Revert entries (newest first) of `run_id`, of the latest run with entries (`last`), or of all runs."""
        state = self.ctx.state
        if last and run_id is None:
            run_id = state.latest_run_id(with_undo=True)
            if run_id is None:
                return UndoReport()
        report = UndoReport()
        deleted_albums: set[str] = set()
        for entry in state.undo_entries(run_id):
            label = f"#{entry['id']} {entry['action']} {entry['src'] or ''} {entry['dst'] or ''}".strip()
            try:
                note = self._revert(entry, dry_run, deleted_albums)
            except Exception as exc:
                self.ctx.log.exception("undo entry %s failed", entry["id"])
                report.errors.append(f"{label}: {type(exc).__name__}: {exc}")
                continue
            if dry_run:
                report.planned.append(label)
                continue
            state.mark_undone([entry["id"]])
            if note == "noop":
                report.skipped += 1
            else:
                report.undone += 1
        return report

    # ------------------------------------------------------------------ per-action handlers
    def _revert(self, entry: dict[str, Any], dry_run: bool, deleted_albums: set[str]) -> str:
        action = entry["action"]
        src, dst, meta = entry["src"], entry["dst"], entry["meta"] or {}
        if action == "move":
            return self._undo_move(src, dst, dry_run)
        if action in ("symlink", "hardlink", "copy", "write_file"):
            return self._undo_remove(dst, dry_run)
        if action == "mkdir":
            return self._undo_mkdir(dst, dry_run)
        if action.startswith("immich_"):
            return self._undo_immich(action, meta, dry_run, deleted_albums)
        raise ValueError(f"未知的撤销动作: {action}")

    @staticmethod
    def _undo_move(src: str, dst: str, dry_run: bool) -> str:
        if not os.path.lexists(dst):
            return "noop"
        if os.path.lexists(src):
            raise FileExistsError(f"原位置已被占用，无法移回: {src}")
        if not dry_run:
            Path(src).parent.mkdir(parents=True, exist_ok=True)
            shutil.move(dst, src)
        return "ok"

    @staticmethod
    def _undo_remove(dst: str, dry_run: bool) -> str:
        if not os.path.lexists(dst):
            return "noop"
        if os.path.isdir(dst) and not os.path.islink(dst):
            raise IsADirectoryError(f"目标是目录，拒绝删除: {dst}")
        if not dry_run:
            os.unlink(dst)
        return "ok"

    @staticmethod
    def _undo_mkdir(dst: str, dry_run: bool) -> str:
        if not os.path.isdir(dst):
            return "noop"
        if os.listdir(dst):
            return "noop"  # not empty: leave it, but the entry counts as handled
        if not dry_run:
            os.rmdir(dst)
        return "ok"

    def _undo_immich(self, action: str, meta: dict[str, Any], dry_run: bool, deleted_albums: set[str]) -> str:
        immich = self.ctx.immich
        if immich is None:
            raise RuntimeError("Immich 未启用，无法撤销 Immich 操作")
        if action == "immich_album":
            album_id = str(meta["album_id"])
            if not dry_run:
                immich.delete_album(album_id)
                deleted_albums.add(album_id)
            return "ok"
        if action == "immich_add_assets":
            return "noop"  # removing the album (a later-logged-first entry) drops its membership
        if action == "immich_tag":
            if not dry_run:
                immich.untag_assets(meta["tag_id"], list(meta.get("asset_ids", [])))
            return "ok"
        if action == "immich_description":
            if not dry_run:
                immich.update_asset(meta["asset_id"], description=meta.get("previous") or "")
            return "ok"
        raise ValueError(f"未知的撤销动作: {action}")
