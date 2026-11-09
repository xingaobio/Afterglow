"""Run context (ARCHITECTURE.md §4): configuration, state, logger and lazily created integration clients."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from afterglow.config import Config
from afterglow.state import State

if TYPE_CHECKING:
    from afterglow.integrations.immich import ImmichClient
    from afterglow.integrations.llm import LLMClient
    from afterglow.integrations.lumen import LumenCLI, LumenLib
    from afterglow.undo import UndoLog
    from afterglow.util.fs import SafeFS


@dataclass
class Context:
    """Everything a stage needs.  Integration clients are created on first use."""

    config: Config
    state: State
    log: logging.Logger = field(default_factory=lambda: logging.getLogger("afterglow"))
    run_id: int = 0
    dry_run: bool = False
    _cache: dict[str, Any] = field(default_factory=dict, repr=False)

    def _get(self, key: str, factory: Any) -> Any:
        if key not in self._cache:
            self._cache[key] = factory()
        return self._cache[key]

    @property
    def lumen(self) -> LumenCLI:
        def make() -> Any:
            from afterglow.integrations.lumen import LumenCLI

            return LumenCLI(self.config.lumen, log=self.log)

        return self._get("lumen", make)

    @property
    def lumen_lib(self) -> LumenLib | None:
        def make() -> Any:
            try:
                from afterglow.integrations.lumen import LumenLib

                lib = LumenLib(self.config.lumen.source_dir)
            except Exception:
                self.log.exception("LumenLib init failed")
                return None
            return lib if getattr(lib, "available", False) else None

        return self._get("lumen_lib", make)

    @property
    def llm(self) -> LLMClient:
        def make() -> Any:
            from afterglow.integrations.llm import LLMClient

            return LLMClient(self.config.llm.server, self.config.llm.vision_model, self.config.llm.timeout_seconds)

        return self._get("llm", make)

    @property
    def brain_llm(self) -> LLMClient:
        """Client for the larger naming model; falls back to the vision client when no brain server is set."""

        def make() -> Any:
            cfg = self.config.llm
            if not cfg.brain_server:
                return self.llm
            from afterglow.integrations.llm import LLMClient

            return LLMClient(cfg.brain_server, cfg.brain_model, cfg.timeout_seconds)

        return self._get("brain_llm", make)

    @property
    def immich(self) -> ImmichClient | None:
        def make() -> Any:
            if not self.config.immich.enabled:
                return None
            from afterglow.integrations.immich import ImmichClient

            return ImmichClient(self.config.immich.url, self.config.immich.api_key)

        return self._get("immich", make)

    @property
    def undo(self) -> UndoLog:
        def make() -> Any:
            from afterglow.undo import UndoLog

            return UndoLog(self)

        return self._get("undo", make)

    @property
    def fs(self) -> SafeFS:
        def make() -> Any:
            from afterglow.util.fs import SafeFS

            return SafeFS(self)

        return self._get("fs", make)

    def close(self) -> None:
        """Release subprocess-backed clients (best effort)."""
        client = self._cache.get("lumen")
        if client is not None:
            try:
                client.close()
            except Exception:
                self.log.exception("closing lumen failed")
