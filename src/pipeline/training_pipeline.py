"""
End-to-end training pipeline.

Run with:
    python -m src.pipeline.training_pipeline
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

from src.components.data_ingestion import ingest_data
from src.components.data_validation import validate_data
from src.components.data_transformation import transform_data
from src.components.model_trainer import train_and_select
from src.config.config import get_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


def run_training_pipeline(csv_path: Path | None = None) -> dict:
    """
    Execute the full training pipeline:
      1. Ingest CSV data
      2. Validate schema & labels
      3. Clean text + TF-IDF vectorization
      4. Train, tune, evaluate, select champion model

    Parameters
    ----------
    csv_path
        Optional override for the dataset CSV path.

    Returns
    -------
    dict
        Summary including champion model name and metrics.
    """
    logger.info("=" * 60)
    logger.info("TRAINING PIPELINE — START")
    logger.info("=" * 60)

    try:
        # 1 — ingest
        logger.info("Phase 1: Data Ingestion")
        df = ingest_data(csv_path)

        # 2 — validate
        logger.info("Phase 2: Data Validation")
        df = validate_data(df)

        # 3 — transform
        logger.info("Phase 3: Data Transformation")
        X_train, y_train, X_test, y_test, vectorizer = transform_data(df)

        # 4 — train + select
        logger.info("Phase 4: Model Training & Selection")
        best_model, best_name, all_metrics = train_and_select(
            X_train, y_train, X_test, y_test
        )

        summary = {
            "champion": best_name,
            "metrics": all_metrics,
            "train_size": X_train.shape[0],
            "test_size": X_test.shape[0],
            "vocab_size": X_train.shape[1],
        }

        logger.info("=" * 60)
        logger.info("TRAINING PIPELINE — COMPLETE")
        logger.info("Champion model: %s", best_name)
        logger.info("=" * 60)
        return summary

    except Exception as exc:
        logger.error("Training pipeline failed: %s", exc)
        logger.debug(traceback.format_exc())
        raise


# ── CLI entry point ────────────────────────────────────────────────────
if __name__ == "__main__":
    csv_override = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    run_training_pipeline(csv_override)
