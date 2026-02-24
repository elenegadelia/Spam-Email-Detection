"""
Central configuration for the Spam Email Detection system.

All paths, hyperparameters, and tunables are defined here so that
every other module imports from a single source of truth.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ── project root (two levels up from this file) ────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class PathConfig:
    """File-system paths used across the project."""

    project_root: Path = PROJECT_ROOT
    dataset_csv: Path = PROJECT_ROOT / "data" / "dataset" / "dataset.csv"
    sample_csv: Path = PROJECT_ROOT / "data" / "sample" / "sample_dataset.csv"
    raw_data_dir: Path = PROJECT_ROOT / "data" / "raw"
    processed_data_dir: Path = PROJECT_ROOT / "data" / "processed"
    models_dir: Path = PROJECT_ROOT / "outputs" / "models"
    vectorizers_dir: Path = PROJECT_ROOT / "outputs" / "vectorizers"
    metrics_dir: Path = PROJECT_ROOT / "outputs" / "metrics"
    logs_dir: Path = PROJECT_ROOT / "logs"

    def ensure_dirs(self) -> None:
        """Create all output / log directories if they don't exist."""
        for d in (
            self.raw_data_dir,
            self.processed_data_dir,
            self.models_dir,
            self.vectorizers_dir,
            self.metrics_dir,
            self.logs_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class TfidfConfig:
    """TF-IDF vectorizer settings."""

    ngram_range: tuple[int, int] = (1, 2)
    max_features: int = 20_000
    min_df: int = 2
    stop_words: str = "english"
    sublinear_tf: bool = True


@dataclass(frozen=True)
class TrainingConfig:
    """Training / evaluation settings."""

    test_size: float = 0.2
    random_seed: int = 42
    cv_folds: int = 5
    scoring_metric: str = "f1"            # used by GridSearchCV
    best_model_metric: str = "f1_spam"    # selects final champion model


@dataclass(frozen=True)
class ModelGrids:
    """Hyperparameter grids for GridSearchCV (kept small for speed)."""

    svm: dict[str, list[Any]] = field(default_factory=lambda: {
        "C": [0.1, 1.0, 10.0],
        "loss": ["hinge", "squared_hinge"],
    })
    logistic_regression: dict[str, list[Any]] = field(default_factory=lambda: {
        "C": [0.1, 1.0, 10.0],
        "solver": ["lbfgs", "liblinear"],
    })


@dataclass(frozen=True)
class Config:
    """Top-level config aggregating all sub-configs."""

    paths: PathConfig = field(default_factory=PathConfig)
    tfidf: TfidfConfig = field(default_factory=TfidfConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    grids: ModelGrids = field(default_factory=ModelGrids)

    def __post_init__(self) -> None:
        self.paths.ensure_dirs()


# Singleton for convenience ─────────────────────────────────────────────
_config: Config | None = None


def get_config() -> Config:
    """Return the global Config singleton (created on first call)."""
    global _config
    if _config is None:
        _config = Config()
    return _config
