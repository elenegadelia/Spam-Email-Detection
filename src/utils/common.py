"""
Shared helper functions: artifact I/O, math utils, text cleaning.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from bs4 import BeautifulSoup

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ── Artifact persistence ───────────────────────────────────────────────

def save_artifact(obj: Any, directory: Path, base_name: str) -> Path:
    """
    Persist *obj* via joblib with a timestamped filename.

    Returns the full path to the saved file.
    """
    directory.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = directory / f"{base_name}_{ts}.joblib"
    joblib.dump(obj, path)
    logger.info("Artifact saved → %s", path)
    return path


def load_artifact(path: Path) -> Any:
    """Load a joblib artifact, raising ``FileNotFoundError`` with a
    helpful message when the file is missing."""
    if not path.exists():
        raise FileNotFoundError(
            f"Artifact not found: {path}\n"
            "Please run the training pipeline first:\n"
            "  python -m src.pipeline.training_pipeline"
        )
    obj = joblib.load(path)
    logger.info("Artifact loaded ← %s", path)
    return obj


def save_json(data: dict, path: Path) -> None:
    """Write a dict to a JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
    logger.info("JSON saved → %s", path)


def load_json(path: Path) -> dict:
    """Read a JSON file into a dict."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ── Math helpers ───────────────────────────────────────────────────────

def sigmoid(x: np.ndarray) -> np.ndarray:
    """Element-wise sigmoid, clipped to avoid overflow."""
    x = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x))


# ── Text cleaning ─────────────────────────────────────────────────────

_HEADER_RE = re.compile(
    r"^(From|To|Cc|Bcc|Subject|Date|Reply-To|Content-Type|MIME-Version|"
    r"Message-ID|Received|Return-Path|X-[\w-]+):\s*.*$",
    re.MULTILINE | re.IGNORECASE,
)
_WHITESPACE_RE = re.compile(r"\s+")


def clean_text(raw: str) -> str:
    """
    Clean a raw email body for ML ingestion.

    Steps:
      1. Strip HTML tags (BeautifulSoup).
      2. Remove common email headers.
      3. Collapse whitespace and lowercase.

    Returns an empty string for falsy / non-string input.
    """
    if not raw or not isinstance(raw, str):
        return ""
    # 1 – strip HTML
    text = BeautifulSoup(raw, "lxml").get_text(separator=" ")
    # 2 – remove header lines
    text = _HEADER_RE.sub("", text)
    # 3 – normalise
    text = _WHITESPACE_RE.sub(" ", text).strip().lower()
    return text


def find_latest_artifact(directory: Path, prefix: str) -> Path | None:
    """Return the most-recently modified ``*.joblib`` matching *prefix*
    inside *directory*, or ``None``."""
    candidates = sorted(
        directory.glob(f"{prefix}_*.joblib"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None
