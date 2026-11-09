"""Inter-process GPU mutex; optionally pauses Immich ML jobs while a GPU stage runs."""

from __future__ import annotations

from types import TracebackType
from typing import TYPE_CHECKING

from filelock import FileLock

if TYPE_CHECKING:
    from afterglow.context import Context

LOCK_TIMEOUT_SECONDS = 3600
PAUSED_JOBS = ("smartSearch", "faceDetection")


class GpuLock:
    def __init__(self, ctx: Context, timeout: float = LOCK_TIMEOUT_SECONDS) -> None:
        self.ctx = ctx
        self._lock = FileLock(str(ctx.config.gpu.lock_file), timeout=timeout)
        self._paused: list[str] = []

    def __enter__(self) -> GpuLock:
        self.ctx.config.gpu.lock_file.parent.mkdir(parents=True, exist_ok=True)
        self._lock.acquire()
        if self.ctx.config.immich.pause_ml_during_gpu_stages and self.ctx.immich is not None:
            for job in PAUSED_JOBS:
                try:
                    self.ctx.immich.pause_job(job)
                    self._paused.append(job)
                except Exception as exc:
                    self.ctx.log.warning("无法暂停 Immich 任务 %s: %s", job, exc)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        try:
            immich = self.ctx.immich
            for job in self._paused:
                try:
                    if immich is not None:
                        immich.resume_job(job)
                except Exception as err:
                    self.ctx.log.warning("无法恢复 Immich 任务 %s: %s", job, err)
        finally:
            self._paused.clear()
            self._lock.release()
