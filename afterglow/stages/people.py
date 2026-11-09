"""Stage `people`: one Immich album per recognised person (ARCHITECTURE.md §11.1).

Global stage, no per-file steps.  Immich's face recognition runs after the external library scan, so the
first run may find nobody; that is normal and the albums appear on a later run.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from typing import TYPE_CHECKING

from afterglow.stages.base import FileRecord, Stage, StepOutcome
from afterglow.stages.publish import ORGANIZED_KEY

if TYPE_CHECKING:
    from afterglow.context import Context

log = logging.getLogger(__name__)


class People(Stage):
    name = "people"
    requires = ("publish",)
    uses_gpu = False
    per_file = False

    def run(self, ctx: Context, batch: list[FileRecord]) -> list[StepOutcome]:
        cfg = ctx.config.people
        if ctx.immich is None or not cfg.enabled or not ctx.config.immich.enabled:
            log.debug("people stage disabled or Immich not configured")
            return []
        if ctx.dry_run:
            log.info("试运行：跳过人物相册")
            return []
        try:
            persons = ctx.immich.people()
        except Exception:
            log.exception("读取 Immich 人物列表失败")
            return []
        if not persons:
            log.info("Immich 暂未识别到人物（人脸识别需在图库扫描后运行一段时间），下次运行再建人物相册")
            return []
        unnamed = 0
        for person in persons:
            name = str(person.get("name") or "").strip()
            pid = str(person.get("id") or "")
            if not pid:
                continue
            if not name:
                if not cfg.include_unnamed:
                    continue
                unnamed += 1
                name = f"未命名 {unnamed}"
            try:
                self._one(ctx, pid, name)
            except Exception:
                log.warning("人物相册处理失败: %s", name, exc_info=True)
        return []

    # ----------------------------------------------------------------------------------------
    @staticmethod
    def _map_file(ctx: Context, path: str) -> FileRecord | None:
        """Immich originalPath -> files row (direct, then through a symlink in the organized tree)."""
        state = ctx.state
        rec = state.file_by_path(path)
        if rec is None:
            file_id = state.get(ORGANIZED_KEY + path)  # a copy placed by publish (organize.mode = copy_picks)
            if file_id and str(file_id).isdigit():
                rec = state.get_file(int(file_id))
        if rec is None:
            try:
                real = os.path.realpath(path)
            except OSError:
                real = path
            if real != path:
                rec = state.file_by_path(real)
        if rec is not None and rec.pair_id:  # a paired JPEG follows its RAW
            main = state.get_file(rec.pair_id) if hasattr(state, "get_file") else None
            rec = main or rec
        return rec

    def _one(self, ctx: Context, pid: str, name: str) -> None:
        cfg, state, immich = ctx.config.people, ctx.state, ctx.immich
        assert immich is not None
        by_file: dict[int, tuple[FileRecord, str]] = {}
        for asset in immich.person_assets(pid):
            asset_id, path = asset.get("id"), asset.get("originalPath")
            if not asset_id or not path:
                continue
            rec = self._map_file(ctx, str(path))
            if rec is not None and rec.id not in by_file:
                by_file[rec.id] = (rec, str(asset_id))
        if len(by_file) < cfg.min_photos:
            log.debug("person %s skipped: %d mapped files < min_photos", name, len(by_file))
            return
        scored: list[tuple[float, int, str]] = []
        for fid, (rec, asset_id) in by_file.items():
            analysis = rec.analysis if rec.analysis is not None else state.analysis(fid)
            if not analysis or not analysis.get("pick"):
                continue
            scored.append((float(analysis.get("quality") or 0.0), fid, asset_id))
        scored.sort(key=lambda t: (-t[0], t[1]))
        if cfg.best_per_person > 0:
            scored = scored[: cfg.best_per_person]
        if not scored:
            log.debug("person %s has no picks", name)
            return
        asset_ids = [a for _, _, a in scored]
        sig = hashlib.sha1("\n".join(sorted(asset_ids)).encode()).hexdigest()
        album_key, sig_key, members_key = (f"immich_person_{k}:{pid}" for k in ("album", "sig", "members"))
        album_name = cfg.album_prefix + name

        if state.get(sig_key) != sig:
            album_id = state.get(album_key)
            if album_id:
                try:
                    known = set(json.loads(state.get(members_key) or "[]"))
                except ValueError:
                    known = set()
                new = [a for a in asset_ids if a not in known]
                if new:
                    ctx.undo.record("immich_add_assets", meta={"album_id": album_id, "asset_ids": new})
                    immich.add_to_album(album_id, new)
            else:
                album_id = immich.create_album(album_name, asset_ids)  # the id exists only after creation
                ctx.undo.record("immich_album", meta={"album_id": album_id, "name": album_name})
                state.set(album_key, album_id)
            state.set(members_key, json.dumps(sorted(asset_ids)))
            state.set(sig_key, sig)
            log.info("人物相册已更新: %s（%d 张）", album_name, len(asset_ids))

        self._local_group(ctx, pid, name, [fid for _, fid, _ in scored])

    @staticmethod
    def _local_group(ctx: Context, pid: str, name: str, file_ids: list[int]) -> None:
        state = ctx.state
        for group in state.groups("person"):
            if (group.get("meta") or {}).get("immich_person_id") == pid:
                fresh = [f for f in file_ids if f not in set(group["members"])]
                if fresh:
                    state.add_group_members(group["id"], fresh)
                return
        state.create_group("person", file_ids, name=name, meta={"immich_person_id": pid})


STAGE = People
