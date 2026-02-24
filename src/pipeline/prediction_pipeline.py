"""
Prediction pipeline: load saved artifacts and classify new emails.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config.config import get_config
from src.utils.common import clean_text, find_latest_artifact, load_artifact, sigmoid
from src.utils.logger import get_logger
from src.utils.mbox_parser import parse_mbox

logger = get_logger(__name__)


class PredictionPipeline:
    """
    Loads the most recent model + vectorizer and exposes ``predict``
    (single text) and ``predict_batch_mbox`` (mbox file) methods.
    """

    def __init__(
        self,
        model_path: Path | None = None,
        vectorizer_path: Path | None = None,
    ) -> None:
        cfg = get_config()

        # Resolve latest artifacts when explicit paths are not given
        self._model_path = model_path or find_latest_artifact(
            cfg.paths.models_dir, "best_model"
        )
        self._vec_path = vectorizer_path or find_latest_artifact(
            cfg.paths.vectorizers_dir, "tfidf_vectorizer"
        )

        if self._model_path is None or self._vec_path is None:
            raise FileNotFoundError(
                "Model or vectorizer artifacts not found.\n"
                "Please run the training pipeline first:\n"
                "  python -m src.pipeline.training_pipeline"
            )

        self.model: Any = load_artifact(self._model_path)
        self.vectorizer: Any = load_artifact(self._vec_path)
        logger.info(
            "PredictionPipeline ready  model=%s  vectorizer=%s",
            self._model_path.name,
            self._vec_path.name,
        )

    # ── helpers ────────────────────────────────────────────────────

    @property
    def model_name(self) -> str:
        return type(self.model).__name__

    def _confidence(self, X) -> np.ndarray:
        """Return a probability-like confidence score in [0, 1] for the
        **spam** class (label = 1).

        * If the model has ``predict_proba`` → use it directly.
        * Otherwise fall back to ``decision_function`` → sigmoid.
        """
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(X)
            return proba[:, 1]  # P(spam)

        if hasattr(self.model, "decision_function"):
            raw = self.model.decision_function(X)
            return sigmoid(np.asarray(raw, dtype=float))

        # Last resort: binary prediction → 0 or 1
        return self.model.predict(X).astype(float)

    # ── public API ─────────────────────────────────────────────────

    def predict(self, text: str) -> dict[str, Any]:
        """
        Classify a single email body.

        Returns
        -------
        dict with keys ``label``, ``confidence``, ``model``.
        """
        cleaned = clean_text(text)
        if not cleaned or cleaned == "[empty]":
            return {"label": "unknown", "confidence": 0.0, "model": self.model_name}

        X = self.vectorizer.transform([cleaned])
        conf = float(self._confidence(X)[0])
        label = "spam" if conf >= 0.5 else "ham"
        return {
            "label": label,
            "confidence": round(conf if label == "spam" else 1 - conf, 4),
            "model": self.model_name,
        }

    def predict_batch_mbox(self, source) -> pd.DataFrame:
        """
        Parse an ``.mbox`` file and classify every message.

        Parameters
        ----------
        source
            Path or file-like object accepted by :func:`parse_mbox`.

        Returns
        -------
        pd.DataFrame
            Columns: email_index, subject, sender, date, predicted_label,
            confidence.
        """
        df = parse_mbox(source)

        cleaned = df["body"].astype(str).apply(clean_text).replace("", "[empty]")
        X = self.vectorizer.transform(cleaned)
        confs = self._confidence(X)

        df["predicted_label"] = np.where(confs >= 0.5, "spam", "ham")
        df["confidence"] = np.round(
            np.where(confs >= 0.5, confs, 1 - confs), 4
        )
        df.drop(columns=["body"], inplace=True)
        logger.info("Batch prediction complete: %d emails.", len(df))
        return df
