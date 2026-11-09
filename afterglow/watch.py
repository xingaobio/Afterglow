"""`afterglow watch`: poll for the source drive, run the pipeline, re-run on change or daily."""

from __future__ import annotations

import os
import signal
import threading
import time
from typing import TYPE_CHECKING

from afterglow import pipeline

if TYPE_CHECKING:
    from afterglow.context import Context

DAILY_SECONDS = 24 * 3600


def source_ready(ctx: Context) -> bool:
    """True when the source is a readable directory that contains at least one entry."""
    src = ctx.config.paths.source
    if not src.is_dir() or not os.access(src, os.R_OK | os.X_OK):
        return False
    try:
        return any(True for _ in src.iterdir())
    except OSError:
        return False


def newest_mtime(ctx: Context) -> float:
    """Newest mtime found while walking the source tree (directories included)."""
    newest = 0.0
    for root, dirs, files in os.walk(ctx.config.paths.source):
        dirs[:] = [d for d in dirs if not d.startswith(".") and d != "@eaDir"]
        for name in [".", *dirs, *files]:
            try:
                newest = max(newest, os.stat(os.path.join(root, name)).st_mtime)
            except OSError:
                continue
    return newest


def watch(ctx: Context, interval: float = 60, stop: threading.Event | None = None) -> int:
    """Block until SIGTERM/SIGINT.  Returns 0 on clean exit."""
    stop = stop or threading.Event()
    log = ctx.log

    def _handler(signum: int, _frame: object) -> None:
        log.info("收到信号 %s，准备退出", signum)
        stop.set()

    previous: dict[int, object] = {}
    if threading.current_thread() is threading.main_thread():
        for sig in (signal.SIGTERM, signal.SIGINT):
            previous[sig] = signal.signal(sig, _handler)
    try:
        last_run = 0.0
        last_mtime = -1.0
        while not stop.is_set():
            if source_ready(ctx):
                mtime = newest_mtime(ctx)
                if last_run == 0.0 or mtime != last_mtime or time.monotonic() - last_run >= DAILY_SECONDS:
                    log.info("开始运行流水线")
                    try:
                        report = pipeline.run(ctx)
                        log.info("流水线完成: 成功 %d，失败 %d，跳过 %d", report.ok, report.failed, report.skipped)
                    except Exception:
                        log.exception("pipeline run failed")
                    last_run = time.monotonic()
                    last_mtime = newest_mtime(ctx)
            else:
                log.info("等待移动硬盘挂载: %s", ctx.config.paths.source)
            stop.wait(interval)
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)  # type: ignore[arg-type]
    log.info("已退出监听")
    return 0
