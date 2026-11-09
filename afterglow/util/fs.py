"""Safe filesystem operations: every change is written to the undo log first; the source tree is never touched."""

from __future__ import annotations

import contextlib
import os
import shutil
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from afterglow.context import Context


class SourceProtectedError(PermissionError):
    """An operation would modify something under `paths.source`."""


def _norm(path: str | os.PathLike[str]) -> Path:
    """Absolute path with the parent directory resolved but the final component left alone (symlink-safe)."""
    p = Path(os.path.abspath(os.fspath(path)))
    return p.parent.resolve() / p.name if p.name else p.resolve()


def is_under(path: str | os.PathLike[str], root: str | os.PathLike[str]) -> bool:
    """True when `path` is `root` or lies below it (without following a final symlink)."""
    p = _norm(path)
    r = Path(root).resolve()
    return p == r or r in p.parents


def atomic_write_bytes(path: str | os.PathLike[str], data: bytes) -> None:
    """Write `data` to `path` through a temporary file in the same directory + `os.replace`."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, target)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def unique_path(path: str | os.PathLike[str], sha: str, length: int = 6) -> Path:
    """Return `path` if free, else `<stem>_<sha prefix><suffix>` (with a numeric counter as last resort)."""
    p = Path(path)
    if not os.path.lexists(p):
        return p
    candidate = p.with_name(f"{p.stem}_{sha[:length]}{p.suffix}")
    n = 2
    while os.path.lexists(candidate):
        candidate = p.with_name(f"{p.stem}_{sha[:length]}_{n}{p.suffix}")
        n += 1
    return candidate


class SafeFS:
    """Undo-logged mkdir / symlink / hardlink / copy / move / write_text bound to a Context."""

    def __init__(self, ctx: Context) -> None:
        self.ctx = ctx

    # ------------------------------------------------------------------ helpers
    @property
    def _source(self) -> Path:
        return self.ctx.config.paths.source

    def _guard_dst(self, dst: Path) -> None:
        if is_under(dst, self._source):
            raise SourceProtectedError(f"不允许写入原图目录: {dst}")

    def _log(self, action: str, src: object, dst: object, meta: dict | None = None) -> None:
        self.ctx.state.log_undo(
            self.ctx.run_id, action, None if src is None else str(src), None if dst is None else str(dst), meta
        )

    def _dry(self, what: str, *args: object) -> bool:
        if self.ctx.dry_run:
            self.ctx.log.info("[dry-run] %s %s", what, " -> ".join(str(a) for a in args))
        return self.ctx.dry_run

    # ------------------------------------------------------------------ operations
    def mkdir(self, path: str | os.PathLike[str]) -> None:
        """Create `path` and missing parents; every directory actually created is logged (top-down)."""
        target = _norm(path)
        self._guard_dst(target)
        missing: list[Path] = []
        cur = target
        while not cur.exists() and cur != cur.parent:
            missing.append(cur)
            cur = cur.parent
        if not missing:
            return
        if self._dry("mkdir", target):
            return
        for d in reversed(missing):
            self._log("mkdir", None, d)
            d.mkdir(exist_ok=True)

    def _prepare(self, dst: Path) -> None:
        self._guard_dst(dst)
        if os.path.lexists(dst):
            raise FileExistsError(f"目标已存在: {dst}")
        self.mkdir(dst.parent)

    def symlink(self, target: str | os.PathLike[str], link: str | os.PathLike[str]) -> None:
        """Create `link` pointing at `target`."""
        dst = _norm(link)
        self._prepare(dst)
        if self._dry("symlink", dst, target):
            return
        self._log("symlink", target, dst)
        os.symlink(os.fspath(target), dst)

    def hardlink(self, src: str | os.PathLike[str], dst: str | os.PathLike[str]) -> None:
        d = _norm(dst)
        self._prepare(d)
        if self._dry("hardlink", src, d):
            return
        self._log("hardlink", src, d)
        os.link(src, d)

    def copy(self, src: str | os.PathLike[str], dst: str | os.PathLike[str]) -> None:
        d = _norm(dst)
        self._prepare(d)
        if self._dry("copy", src, d):
            return
        self._log("copy", src, d)
        shutil.copy2(src, d)

    def move(
        self,
        src: str | os.PathLike[str],
        dst: str | os.PathLike[str],
        allow_move_from_source: bool = False,
    ) -> None:
        """Move `src` to `dst`.  Moving out of the source tree needs `allow_move_from_source=True` (publish only)."""
        s = _norm(src)
        d = _norm(dst)
        if is_under(s, self._source) and not allow_move_from_source:
            raise SourceProtectedError(f"不允许移动原图: {s}")
        self._prepare(d)
        if self._dry("move", s, d):
            return
        self._log("move", s, d)
        shutil.move(os.fspath(s), os.fspath(d))

    def write_text(self, path: str | os.PathLike[str], text: str, encoding: str = "utf-8") -> None:
        self.write_bytes(path, text.encode(encoding))

    def write_bytes(self, path: str | os.PathLike[str], data: bytes) -> None:
        dst = _norm(path)
        self._guard_dst(dst)
        self.mkdir(dst.parent)
        if self._dry("write", dst):
            return
        self._log("write_file", None, dst, {"overwrote": dst.exists()})
        atomic_write_bytes(dst, data)
