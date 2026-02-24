"""
Structured rotating-file + console logger for the project.

Usage:
    from src.utils.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Training started")
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path

from src.config.config import get_config

_INITIALISED: set[str] = set()


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Return a logger that writes to *both* the console and a rotating log
    file under ``logs/``.

    Parameters
    ----------
    name : str
        Typically ``__name__`` of the calling module.
    level : int
        Minimum severity level.

    Returns
    -------
    logging.Logger
    """
    if name in _INITIALISED:
        return logging.getLogger(name)

    cfg = get_config()
    log_dir = cfg.paths.logs_dir
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / f"spam_detector_{datetime.now():%Y%m%d}.log"

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Rotating file handler (5 MB × 3 backups)
    fh = RotatingFileHandler(
        log_file, maxBytes=5_000_000, backupCount=3, encoding="utf-8"
    )
    fh.setLevel(level)
    fh.setFormatter(formatter)

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.propagate = False

    _INITIALISED.add(name)
    return logger
