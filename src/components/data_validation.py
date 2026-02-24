"""
Data validation: ensure the dataset schema is correct before training.
"""

from __future__ import annotations

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

REQUIRED_COLUMNS = {"text", "label"}
VALID_LABELS = {"spam", "ham"}


def validate_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate and normalise the raw dataset.

    Checks
    ------
    * Required columns ``text`` and ``label`` exist.
    * ``label`` values are "spam" or "ham" (case-insensitive).
    * Drops rows with missing ``text`` or ``label``.

    Returns
    -------
    pd.DataFrame
        Cleaned copy of the input.

    Raises
    ------
    ValueError
        On schema violations.
    """
    # ── column check ───────────────────────────────────────────────
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"Dataset is missing required columns: {missing}. "
            f"Found columns: {list(df.columns)}"
        )

    df = df.copy()

    # ── drop missing rows ─────────────────────────────────────────
    before = len(df)
    df.dropna(subset=["text", "label"], inplace=True)
    dropped = before - len(df)
    if dropped:
        logger.warning("Dropped %d rows with missing text/label.", dropped)

    # ── normalise labels ──────────────────────────────────────────
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    invalid = set(df["label"].unique()) - VALID_LABELS
    if invalid:
        raise ValueError(
            f"Invalid label values found: {invalid}. "
            f"Labels must be 'spam' or 'ham'."
        )

    # ── log class balance ─────────────────────────────────────────
    counts = df["label"].value_counts()
    logger.info("Class distribution after validation:\n%s", counts.to_string())
    return df.reset_index(drop=True)
