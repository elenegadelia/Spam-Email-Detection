"""
Model trainer: train, tune, evaluate, and persist the best model.

Supports: LinearSVC, LogisticRegression, DecisionTreeClassifier,
          RandomForestClassifier.
GridSearchCV is applied to SVM and LogisticRegression.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from scipy.sparse import spmatrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from src.config.config import get_config
from src.utils.common import save_artifact, save_json
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ── candidate definitions ──────────────────────────────────────────────

def _build_candidates(seed: int) -> dict[str, Any]:
    """Return a dict of model_name → estimator (not yet fitted)."""
    return {
        "LinearSVC": LinearSVC(random_state=seed, max_iter=5000),
        "LogisticRegression": LogisticRegression(
            random_state=seed, max_iter=1000
        ),
        "DecisionTree": DecisionTreeClassifier(random_state=seed),
        "RandomForest": RandomForestClassifier(
            n_estimators=100, random_state=seed, n_jobs=-1
        ),
    }


# ── evaluation helper ──────────────────────────────────────────────────

def _evaluate(
    name: str,
    model: Any,
    X_test: spmatrix,
    y_test: np.ndarray,
) -> dict[str, float]:
    """Compute and return evaluation metrics for *model* on the test set."""
    y_pred = model.predict(X_test)
    metrics = {
        "model": name,
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision_spam": round(precision_score(y_test, y_pred, pos_label=1, zero_division=0), 4),
        "recall_spam": round(recall_score(y_test, y_pred, pos_label=1, zero_division=0), 4),
        "f1_spam": round(f1_score(y_test, y_pred, pos_label=1, zero_division=0), 4),
        "f1_macro": round(f1_score(y_test, y_pred, average="macro", zero_division=0), 4),
    }
    report = classification_report(y_test, y_pred, target_names=["ham", "spam"])
    logger.info("── %s ──\n%s", name, report)
    return metrics


# ── main entry point ───────────────────────────────────────────────────

def train_and_select(
    X_train: spmatrix,
    y_train: np.ndarray,
    X_test: spmatrix,
    y_test: np.ndarray,
) -> tuple[Any, str, list[dict[str, Any]]]:
    """
    Train all candidate models, apply GridSearchCV where configured,
    evaluate on the hold-out test set, pick the champion, and save it.

    Returns
    -------
    best_model, best_model_name, all_metrics
    """
    cfg = get_config()
    seed = cfg.training.random_seed
    cv = StratifiedKFold(n_splits=cfg.training.cv_folds, shuffle=True, random_state=seed)

    candidates = _build_candidates(seed)
    grids: dict[str, dict] = {
        "LinearSVC": cfg.grids.svm,
        "LogisticRegression": cfg.grids.logistic_regression,
    }

    fitted: dict[str, Any] = {}
    all_metrics: list[dict[str, Any]] = []

    for name, estimator in candidates.items():
        logger.info("Training %s …", name)
        if name in grids:
            gs = GridSearchCV(
                estimator,
                grids[name],
                cv=cv,
                scoring=cfg.training.scoring_metric,
                n_jobs=-1,
                refit=True,
            )
            gs.fit(X_train, y_train)
            logger.info(
                "%s best params: %s  (CV %s=%.4f)",
                name,
                gs.best_params_,
                cfg.training.scoring_metric,
                gs.best_score_,
            )
            fitted[name] = gs.best_estimator_
        else:
            estimator.fit(X_train, y_train)
            fitted[name] = estimator

        metrics = _evaluate(name, fitted[name], X_test, y_test)
        all_metrics.append(metrics)

    # ── select champion ───────────────────────────────────────────
    best_metric_key = cfg.training.best_model_metric  # "f1_spam"
    champion_metrics = max(all_metrics, key=lambda m: m[best_metric_key])
    champion_name = champion_metrics["model"]
    champion_model = fitted[champion_name]

    logger.info(
        "🏆 Champion: %s  (%s = %.4f)",
        champion_name,
        best_metric_key,
        champion_metrics[best_metric_key],
    )

    # ── persist model ─────────────────────────────────────────────
    save_artifact(champion_model, cfg.paths.models_dir, f"best_model_{champion_name}")

    # ── persist metrics ───────────────────────────────────────────
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics_path = cfg.paths.metrics_dir / f"metrics_{ts}.json"
    save_json({"champion": champion_name, "results": all_metrics}, metrics_path)

    return champion_model, champion_name, all_metrics
