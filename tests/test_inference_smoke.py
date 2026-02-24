"""
End-to-end smoke test: train on the sample dataset, then predict.

This test is intentionally small so it runs in seconds.
Works both with ``pytest`` and as a standalone script.
"""

from __future__ import annotations

try:
    import pytest
    HAS_PYTEST = True
except ImportError:
    HAS_PYTEST = False

from src.config.config import get_config
from src.pipeline.training_pipeline import run_training_pipeline
from src.pipeline.prediction_pipeline import PredictionPipeline


# ── shared setup ───────────────────────────────────────────────────────

_pipeline_cache: PredictionPipeline | None = None


def _get_pipeline() -> PredictionPipeline:
    """Train on the tiny sample dataset (once) and return a pipeline."""
    global _pipeline_cache
    if _pipeline_cache is None:
        cfg = get_config()
        run_training_pipeline(cfg.paths.sample_csv)
        _pipeline_cache = PredictionPipeline()
    return _pipeline_cache


if HAS_PYTEST:
    @pytest.fixture(scope="module")
    def trained_pipeline() -> PredictionPipeline:
        return _get_pipeline()


# ── test class ─────────────────────────────────────────────────────────

class TestInferenceSmoke:
    """Smoke tests for the prediction pipeline."""

    def _pipe(self, trained_pipeline=None) -> PredictionPipeline:
        return trained_pipeline if trained_pipeline else _get_pipeline()

    def test_spam_classified(self, trained_pipeline=None) -> None:
        pipe = self._pipe(trained_pipeline)
        result = pipe.predict(
            "Congratulations! You won a free cruise. Click here NOW to claim."
        )
        assert result["label"] in ("spam", "ham")
        assert 0 <= result["confidence"] <= 1

    def test_ham_classified(self, trained_pipeline=None) -> None:
        pipe = self._pipe(trained_pipeline)
        result = pipe.predict(
            "Hey, can we reschedule our meeting to 3 PM tomorrow?"
        )
        assert result["label"] in ("spam", "ham")
        assert 0 <= result["confidence"] <= 1

    def test_empty_text_returns_unknown(self, trained_pipeline=None) -> None:
        pipe = self._pipe(trained_pipeline)
        result = pipe.predict("")
        assert result["label"] == "unknown"
        assert result["confidence"] == 0.0

    def test_model_name_present(self, trained_pipeline=None) -> None:
        pipe = self._pipe(trained_pipeline)
        result = pipe.predict("Hello world")
        assert len(result["model"]) > 0
