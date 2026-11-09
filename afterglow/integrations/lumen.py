"""LUMEN RAW integration: a stdio MCP black box (``LumenCLI``) and optional direct imports (``LumenLib``)."""

from __future__ import annotations

import base64
import contextlib
import json
import logging
import os
import queue
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from typing import Any

import numpy as np

from afterglow.config import Lumen

_log = logging.getLogger(__name__)

PROTOCOL_VERSION = "2025-06-18"
DEFAULT_TIMEOUT = 600.0
_EOF = object()


class LumenError(RuntimeError):
    """LUMEN reported an error, died, or timed out."""


class _ProcessDied(LumenError):
    """The subprocess exited before answering."""


class LumenCLI:
    """``main.py --cli mcp --headless`` subprocess speaking newline-delimited JSON-RPC 2.0 on stdio.

    Edits live inside the server process between calls (open -> auto -> export), so an automatic restart
    discards them. Wrap multi-call sequences in ``with cli.sequence():`` to defer count-based restarts.
    """

    def __init__(self, cfg: Lumen, log: logging.Logger | None = None, timeout: float = DEFAULT_TIMEOUT) -> None:
        self.cfg = cfg
        self.log = log or _log
        self.timeout = timeout
        self.tools: set[str] = set()
        self.generation = 0  # incremented on every (re)start
        self._lock = threading.RLock()
        self._proc: subprocess.Popen[bytes] | None = None
        self._inbox: queue.Queue[Any] = queue.Queue()
        self._next_id = 0
        self._calls = 0
        self._sequence_depth = 0

    # -- process management ------------------------------------------------------------------

    def _command(self) -> list[str]:
        """Command line of the server process (overridden in tests)."""
        return [str(self.cfg.python), str(self.cfg.source_dir / "main.py"), "--cli", "mcp", "--headless"]

    def _env(self) -> dict[str, str]:
        env = dict(os.environ)
        env.update(
            LUMEN_MODELS=str(self.cfg.models_dir),
            LUMEN_LIBRARY=str(self.cfg.library_db),
            LUMEN_COMPUTE=str(self.cfg.compute),
            ORT_DISABLE_TELEMETRY="1",
            PYTHONUNBUFFERED="1",
        )
        return env

    def _start(self) -> None:
        self.log.info("Starting LUMEN subprocess")
        try:
            proc = subprocess.Popen(  # noqa: S603 - fixed argv, no shell
                self._command(),
                cwd=str(self.cfg.source_dir),
                env=self._env(),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except OSError as exc:
            raise LumenError(f"cannot start LUMEN: {exc}") from exc
        self._proc = proc
        self._inbox = queue.Queue()
        self.generation += 1
        self._calls = 0
        threading.Thread(target=self._read_stdout, args=(proc, self._inbox), daemon=True).start()
        threading.Thread(target=self._read_stderr, args=(proc,), daemon=True).start()
        try:
            self._handshake()
        except Exception:
            self._stop()
            raise

    def _read_stdout(self, proc: subprocess.Popen[bytes], inbox: queue.Queue[Any]) -> None:
        assert proc.stdout is not None
        try:
            for raw in proc.stdout:
                line = raw.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                try:
                    inbox.put(json.loads(line))
                except ValueError:
                    self.log.warning("LUMEN stdout (non-JSON): %s", line[:200])
        finally:
            inbox.put(_EOF)

    def _read_stderr(self, proc: subprocess.Popen[bytes]) -> None:
        assert proc.stderr is not None
        for raw in proc.stderr:
            line = raw.decode("utf-8", errors="replace").rstrip()
            if line:
                self.log.debug("lumen: %s", line)

    def _alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def _stop(self) -> None:
        proc, self._proc = self._proc, None
        if proc is None:
            return
        with contextlib.suppress(OSError):
            if proc.stdin:
                proc.stdin.close()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        for stream in (proc.stdout, proc.stderr):
            with contextlib.suppress(OSError):
                if stream:
                    stream.close()

    def close(self) -> None:
        """Stop the subprocess gracefully: close stdin, wait 5 s, terminate, then kill."""
        with self._lock:
            self._stop()

    def __enter__(self) -> LumenCLI:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @contextlib.contextmanager
    def sequence(self) -> Iterator[None]:
        """Hold the lock and suppress count-based restarts for a multi-call edit sequence."""
        with self._lock:
            self._sequence_depth += 1
            try:
                yield
            finally:
                self._sequence_depth -= 1

    # -- JSON-RPC ------------------------------------------------------------------------------

    def _send(self, message: dict[str, Any]) -> None:
        if not self._alive() or self._proc is None or self._proc.stdin is None:
            raise LumenError("LUMEN process is not running")
        data = (json.dumps(message, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
        try:
            self._proc.stdin.write(data)
            self._proc.stdin.flush()
        except (OSError, ValueError) as exc:
            raise _ProcessDied(f"cannot write to LUMEN: {exc}") from exc

    def _request(self, method: str, params: dict[str, Any] | None = None, timeout: float | None = None) -> Any:
        self._next_id += 1
        ident = self._next_id
        message: dict[str, Any] = {"jsonrpc": "2.0", "id": ident, "method": method}
        if params is not None:
            message["params"] = params
        self._send(message)
        deadline = time.monotonic() + (timeout or self.timeout)
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self._stop()  # state unknown after a timeout; next call restarts
                raise LumenError(f"LUMEN timed out on {method} after {timeout or self.timeout:.0f}s")
            try:
                msg = self._inbox.get(timeout=remaining)
            except queue.Empty:
                continue
            if msg is _EOF:
                self._inbox.put(_EOF)
                code = self._proc.poll() if self._proc else None
                raise _ProcessDied(f"LUMEN process exited during {method} (code {code})")
            if not isinstance(msg, dict) or msg.get("id") != ident:
                continue  # notifications / stale responses
            if "error" in msg:
                err = msg["error"] or {}
                raise LumenError(f"LUMEN {method} failed: {err.get('message', err)}")
            return msg.get("result")

    def _handshake(self) -> None:
        self._request(
            "initialize",
            {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "afterglow", "version": "0.1.0"},
            },
            timeout=120,
        )
        self._send({"jsonrpc": "2.0", "method": "notifications/initialized"})
        result = self._request("tools/list", {}, timeout=120) or {}
        self.tools = {t.get("name", "") for t in result.get("tools", [])}

    def call_tool(self, name: str, arguments: dict[str, Any] | None = None, timeout: float | None = None) -> Any:
        """Call an MCP tool; returns structuredContent, else parsed JSON text, else raw text."""
        with self._lock:
            try:
                result = self._call_once(name, arguments, timeout)
            except _ProcessDied as exc:
                if self._sequence_depth:
                    raise
                self.log.warning("LUMEN died during %s (%s); restarting and retrying once", name, exc)
                result = self._call_once(name, arguments, timeout)
        return self._unwrap(name, result)

    def _call_once(self, name: str, arguments: dict[str, Any] | None, timeout: float | None) -> Any:
        restart = not self._alive()
        if not restart and self._sequence_depth == 0 and self._calls >= max(1, self.cfg.restart_every):
            self.log.info("Restarting LUMEN after %d calls", self._calls)
            restart = True
        if restart:
            self._stop()
            self._start()
        self._calls += 1
        return self._request("tools/call", {"name": name, "arguments": arguments or {}}, timeout)

    @staticmethod
    def _unwrap(name: str, result: Any) -> Any:
        if not isinstance(result, dict):
            return result
        content = result.get("content") or []
        text = ""
        if content and isinstance(content[0], dict):
            text = str(content[0].get("text", ""))
        if result.get("isError"):
            raise LumenError(text or f"LUMEN tool {name} failed")
        if result.get("structuredContent") is not None:
            return result["structuredContent"]
        if not text:
            return {}
        try:
            return json.loads(text)
        except ValueError:
            return text

    def run_command(self, command: str, args: dict[str, Any], timeout: float | None = None) -> Any:
        return self.call_tool("run_command", {"command": command, "args": args}, timeout)

    # -- LUMEN operations ----------------------------------------------------------------------

    def analyze(self, paths: list[str]) -> dict:
        """catalog.analyze -> {count, photos[], bursts[]}"""
        return self.call_tool("analyze_photos", {"paths": list(paths), "limit": min(2000, max(1, len(paths)))})

    def open(self, path: str) -> dict:
        return self.call_tool("open_photo", {"path": path})

    def auto_tone(self, path: str) -> dict:
        return self.call_tool("auto_tone", {"path": path})

    def apply_preset(self, path: str, name: str, amount: int = 100) -> dict:
        return self.run_command("edit.preset", {"path": path, "name": name, "amount": amount})

    def apply_edits(self, path: str, changes: dict, amount: int = 100, label: str = "") -> dict:
        args: dict[str, Any] = {"path": path, "changes": changes, "amount": max(0, min(150, int(amount)))}
        if label:
            args["label"] = label
        return self.call_tool("apply_edits", args)

    def export(self, path: str, output: str, quality: int, long_edge: int | None, overwrite: bool = True) -> dict:
        args: dict[str, Any] = {"path": path, "output": output, "quality": quality, "overwrite": overwrite}
        if long_edge:
            args["long_edge"] = long_edge
        return self.run_command("photo.export", args)

    def export_batch(self, paths: list[str], folder: str, fmt: str, quality: int, long_edge: int | None) -> dict:
        args: dict[str, Any] = {"paths": list(paths), "folder": folder, "format": fmt, "quality": quality}
        if long_edge:
            args["long_edge"] = long_edge
        return self.call_tool("export_photos", args)

    def timelapse(self, paths: list[str], output: str, fmt: str = "h264", fps: int = 24) -> dict:
        return self.run_command(
            "video.timelapse", {"paths": list(paths), "output": output, "format": fmt, "fps": fps}
        )

    def write_xmp(self, paths: list[str]) -> dict:
        return self.run_command("library.write_xmp", {"paths": list(paths)})

    def info(self, path: str) -> dict:
        return self.call_tool("photo_info", {"path": path})

    def get_edits(self, path: str) -> dict:
        """edit.get -> {path, process, state: {adjustments: {...}, ...}, changes: [...]} (current slider values)."""
        return self.call_tool("get_edits", {"path": path})

    def render_preview(self, path: str, size: int = 1024) -> bytes:
        """render.preview -> JPEG bytes of the photo with its current edits (long edge `size`, 128-2048)."""
        size = max(128, min(2048, int(size)))
        with self._lock:
            try:
                result = self._call_once("render_preview", {"path": path, "size": size}, None)
            except _ProcessDied:
                if self._sequence_depth:
                    raise
                result = self._call_once("render_preview", {"path": path, "size": size}, None)
        if isinstance(result, dict):
            for part in result.get("content") or []:
                if isinstance(part, dict) and part.get("type") == "image" and part.get("data"):
                    if result.get("isError"):
                        break
                    return base64.b64decode(part["data"])
        self._unwrap("render_preview", result)  # raises LumenError when isError
        raise LumenError("LUMEN render_preview returned no image")


class LumenLib:
    """Direct import of LUMEN's culling / straighten / restoration / merge modules (optional)."""

    available: bool

    def __init__(self, source_dir: Any) -> None:
        self.available = False
        self._culling = self._straighten = self._restoration = self._merge = None
        try:
            path = str(source_dir)
            if path not in sys.path:
                sys.path.insert(0, path)
            from lumen import culling, merge, restoration, straighten

            self._culling, self._straighten, self._restoration, self._merge = culling, straighten, restoration, merge
            self.available = True
        except Exception as exc:  # ImportError, or a broken optional dependency inside lumen
            _log.warning("LUMEN library not importable (%s); AI enhance/merge fall back", exc)

    def _need(self) -> None:
        if not self.available:
            raise LumenError("LUMEN library is not available")

    @staticmethod
    def _to_float(rgb: np.ndarray) -> np.ndarray:
        """LUMEN's restoration/merge modules work on float32 RGB in 0..1."""
        return np.ascontiguousarray(np.asarray(rgb, dtype=np.float32) / 255.0)

    @staticmethod
    def _to_uint8(out: Any) -> np.ndarray:
        """LUMEN returns (image, info) tuples; the image is float32 0..1."""
        arr = np.asarray(out[0] if isinstance(out, tuple) else out)
        if arr.dtype == np.uint8:
            return arr
        return np.clip(np.rint(arr.astype(np.float32) * 255.0), 0, 255).astype(np.uint8)

    def analyse(self, rgb: np.ndarray) -> dict:
        self._need()
        return self._culling.analyse(np.ascontiguousarray(rgb))

    def straighten(self, rgb: np.ndarray) -> tuple[float | None, float]:
        self._need()
        angle, confidence = self._straighten.estimate(np.ascontiguousarray(rgb))
        return (None if angle is None else float(angle)), float(confidence)

    def denoise(self, rgb: np.ndarray, amount: int = 70) -> np.ndarray:
        self._need()
        return self._to_uint8(self._restoration.denoise(self._to_float(rgb), amount))

    def super_resolution(self, rgb: np.ndarray, scale: int = 2) -> np.ndarray:
        self._need()
        return self._to_uint8(self._restoration.super_resolution(self._to_float(rgb), scale))

    def merge_images(self, images: list[np.ndarray], kind: str) -> np.ndarray:
        """kind in hdr | focus | pano (LUMEN calls the last one 'panorama')."""
        self._need()
        lumen_kind = "panorama" if kind == "pano" else kind
        frames = [self._to_float(im) for im in images]
        return self._to_uint8(self._merge.merge_images(frames, lumen_kind, reference=0))
