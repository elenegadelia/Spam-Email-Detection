"""
Data ingestion: load the raw CSV dataset into a pandas DataFrame.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config.config import get_config
from src.utils.logger import get_logger

logger = get_logger(__name__)



def ingest_data(csv_path: Path | None = None) -> pd.DataFrame:
    """
    Read the email dataset CSV.

    Parameters
    ----------
    csv_path
        Override path; defaults to ``Config.paths.dataset_csv``.
        Falls back to the sample dataset if the primary path doesn't exist.

    Returns
    -------
    pd.DataFrame
        Must contain at least ``text`` and ``label`` columns.

    Raises
    ------
    FileNotFoundError
        If neither the configured dataset nor the sample exists.
    """
    cfg = get_config()
    path = csv_path or cfg.paths.dataset_csv

    if not path.exists():
        logger.warning("Dataset not found at %s — trying sample dataset.", path)
        path = cfg.paths.sample_csv

    if not path.exists():
        raise FileNotFoundError(
            f"No dataset found. Place a CSV with 'text' and 'label' columns at "
            f"{cfg.paths.dataset_csv} or {cfg.paths.sample_csv}"
        )

    df = pd.read_csv(path, encoding="utf-8")
    logger.info("Loaded %d rows from %s", len(df), path)
    return df
