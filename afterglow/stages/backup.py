"""Stage 1 - rsync backup of the source drive plus verification (ARCHITECTURE.md §5 ①)."""

from __future__ import annotations

import logging
import os
import random
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from afterglow.stages.base import FileRecord, Stage, StepOutcome
from afterglow.util.hashing import sha256_file

if TYPE_CHECKING:
    from afterglow.context import Context

log = logging.getLogger(__name__)

RSYNC_TIMEOUT = 48 * 3600
SAMPLE_RATIO = 0.01
SAMPLE_MIN = 50
EXCLUDED_DIRS = {"lost+found"}


def _walk_files(root: Path) -> dict[str, int]:
    """{relative path: size} of regular files under `root`, skipping lost+found."""
    found: dict[str, int] = {}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in EXCLUDED_DIRS]
        for name in filenames:
            full = Path(dirpath) / name
            try:
                if full.is_symlink() or not full.is_file():
                    continue
                found[str(full.relative_to(root))] = full.stat().st_size
            except OSError:
                log.warning("无法读取文件信息: %s", full)
    return found


class Backup(Stage):
    name = "backup"
    requires = ()
    uses_gpu = False
    per_file = False

    def run(self, ctx: Context, batch: list[FileRecord]) -> list[StepOutcome]:
        cfg = ctx.config
        backup_root = cfg.paths.backup
        if backup_root is None:
            log.info("未配置 paths.backup，跳过备份")
            self._mark(ctx, "skipped")
            return []
        source = cfg.paths.source
        dest = Path(backup_root) / source.name
        if ctx.dry_run:
            log.info("[dry-run] rsync %s/ -> %s/", source, dest)
            return []
        dest.mkdir(parents=True, exist_ok=True)
        cmd = [
            cfg.backup.tool, "-a", "--info=progress2", "--exclude", "lost+found", f"{source}/", f"{dest}/",
        ]
        log.info("开始备份: %s -> %s", source, dest)
        try:
            proc = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=RSYNC_TIMEOUT)
        except FileNotFoundError as exc:
            raise RuntimeError(f"备份工具不可用: {cfg.backup.tool}") from exc
        if proc.returncode not in (0, 24):  # 24 = some source files vanished while copying
            raise RuntimeError(f"rsync 失败 (返回码 {proc.returncode}): {(proc.stderr or '').strip()[:500]}")
        problems = self._verify(ctx, source, dest)
        if problems:
            raise RuntimeError("备份校验失败: " + "; ".join(problems[:5]))
        ctx.state.set("last_backup_ok", datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"))
        log.info("备份完成并通过校验 (%s)", cfg.backup.verify)
        self._mark(ctx, "done")
        return []

    @staticmethod
    def _mark(ctx: Context, status: str) -> None:
        outcomes = [StepOutcome(f.id, status) for f in ctx.state.files_where(missing=False)]  # type: ignore[arg-type]
        ctx.state.set_steps(outcomes, "backup")

    @staticmethod
    def _verify(ctx: Context, source: Path, dest: Path) -> list[str]:
        src = _walk_files(source)
        dst = _walk_files(dest)
        problems: list[str] = []
        missing = [rel for rel in src if rel not in dst]
        if missing:
            problems.append(f"备份缺少 {len(missing)} 个文件，例如 {missing[0]}")
        mismatched = [rel for rel, size in src.items() if rel in dst and dst[rel] != size]
        if mismatched:
            problems.append(f"{len(mismatched)} 个文件大小不一致，例如 {mismatched[0]}")
        if ctx.config.backup.verify == "sha256" and not problems:
            rels = sorted(src)
            count = min(len(rels), max(SAMPLE_MIN, int(len(rels) * SAMPLE_RATIO)))
            for rel in random.Random(0).sample(rels, count):
                if sha256_file(source / rel) != sha256_file(dest / rel):
                    problems.append(f"sha256 不一致: {rel}")
        return problems


STAGE = Backup
