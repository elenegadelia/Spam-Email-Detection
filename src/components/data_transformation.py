"""
Data transformation: text cleaning + TF-IDF vectorization.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import spmatrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from src.config.config import get_config
from src.utils.common import clean_text, save_artifact
from src.utils.logger import get_logger

logger = get_logger(__name__)


def preprocess_texts(series: pd.Series) -> pd.Series:
    """Apply ``clean_text`` to every entry, replacing empties with a
    placeholder so the vectorizer never sees a blank document."""
    cleaned = series.astype(str).apply(clean_text)
    cleaned = cleaned.replace("", "[empty]")
    logger.info("Preprocessed %d documents.", len(cleaned))
    return cleaned


def transform_data(
    df: pd.DataFrame,
) -> tuple[spmatrix, np.ndarray, spmatrix, np.ndarray, TfidfVectorizer]:
    """
    Clean text, fit a TF-IDF vectorizer, and split into train / test.

    Returns
    -------
    X_train, y_train, X_test, y_test, vectorizer
    """
    cfg = get_config()

    # ── clean ──────────────────────────────────────────────────────
    df = df.copy()
    df["clean_text"] = preprocess_texts(df["text"])

    # ── encode labels ─────────────────────────────────────────────
    labels = (df["label"] == "spam").astype(int).values  # 1=spam, 0=ham

    # ── split (stratified, deterministic) ─────────────────────────
    X_train_txt, X_test_txt, y_train, y_test = train_test_split(
        df["clean_text"],
        labels,
        test_size=cfg.training.test_size,
        random_state=cfg.training.random_seed,
        stratify=labels,
    )
    logger.info(
        "Split → train=%d  test=%d  (seed=%d)",
        len(X_train_txt),
        len(X_test_txt),
        cfg.training.random_seed,
    )

    # ── TF-IDF ────────────────────────────────────────────────────
    vectorizer = TfidfVectorizer(
        ngram_range=cfg.tfidf.ngram_range,
        max_features=cfg.tfidf.max_features,
        min_df=cfg.tfidf.min_df,
        stop_words=cfg.tfidf.stop_words,
        sublinear_tf=cfg.tfidf.sublinear_tf,
    )
    X_train = vectorizer.fit_transform(X_train_txt)
    X_test = vectorizer.transform(X_test_txt)
    logger.info("TF-IDF vocabulary size: %d", len(vectorizer.vocabulary_))

    # ── save vectorizer ───────────────────────────────────────────
    save_artifact(vectorizer, cfg.paths.vectorizers_dir, "tfidf_vectorizer")

    return X_train, y_train, X_test, y_test, vectorizer
