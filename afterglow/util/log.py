"""Logging setup: rich console + rotating daily file under paths.logs."""

from __future__ import annotations

import logging
import logging.handlers
from datetime import datetime

from rich.logging import RichHandler

from afterglow.config import Config

_FMT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"


def setup(config: Config, verbose: bool = False) -> logging.Logger:
    """Configure the `afterglow` logger (idempotent) and return it."""
    logger = logging.getLogger("afterglow")
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    console = RichHandler(rich_tracebacks=True, show_path=False, markup=False)
    console.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.addHandler(console)

    try:
        config.paths.logs.mkdir(parents=True, exist_ok=True)
        path = config.paths.logs / f"afterglow-{datetime.now():%Y%m%d}.log"
        fh = logging.handlers.RotatingFileHandler(path, maxBytes=10_000_000, backupCount=5, encoding="utf-8")
        fh.setFormatter(logging.Formatter(_FMT))
        fh.setLevel(logging.DEBUG)
        logger.addHandler(fh)
    except OSError as exc:
        logger.warning("无法创建日志文件: %s", exc)
    return logger
