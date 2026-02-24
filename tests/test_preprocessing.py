"""
Smoke tests for text preprocessing and data validation.
"""

from __future__ import annotations

import pandas as pd

try:
    import pytest
except ImportError:
    pytest = None  # type: ignore[assignment]

from src.utils.common import clean_text
from src.components.data_validation import validate_data


# ── clean_text ─────────────────────────────────────────────────────────

class TestCleanText:
    def test_strips_html_tags(self) -> None:
        raw = "<html><body><p>Hello <b>World</b></p></body></html>"
        result = clean_text(raw)
        assert "<" not in result
        assert "hello" in result
        assert "world" in result

    def test_lowercases(self) -> None:
        assert clean_text("HELLO WORLD") == "hello world"

    def test_collapses_whitespace(self) -> None:
        result = clean_text("hello   \n\t   world")
        assert result == "hello world"

    def test_removes_email_headers(self) -> None:
        raw = "From: alice@example.com\nSubject: Test\n\nActual body text."
        result = clean_text(raw)
        assert "actual body text" in result
        # Header values should be gone
        assert "alice@example.com" not in result

    def test_empty_input(self) -> None:
        assert clean_text("") == ""
        assert clean_text(None) == ""  # type: ignore[arg-type]

    def test_non_string_input(self) -> None:
        assert clean_text(12345) == ""  # type: ignore[arg-type]


# ── validate_data ──────────────────────────────────────────────────────

class TestValidateData:
    def test_valid_df(self) -> None:
        df = pd.DataFrame({
            "text": ["hello", "world"],
            "label": ["spam", "ham"],
        })
        result = validate_data(df)
        assert len(result) == 2
        assert set(result["label"]) == {"spam", "ham"}

    def test_case_insensitive_labels(self) -> None:
        df = pd.DataFrame({
            "text": ["a", "b"],
            "label": ["SPAM", "Ham"],
        })
        result = validate_data(df)
        assert set(result["label"]) == {"spam", "ham"}

    def test_missing_column_raises(self) -> None:
        df = pd.DataFrame({"text": ["x"], "wrong": ["y"]})
        if pytest:
            with pytest.raises(ValueError, match="missing required columns"):
                validate_data(df)
        else:
            try:
                validate_data(df)
                raise AssertionError("Expected ValueError")
            except ValueError as e:
                assert "missing required columns" in str(e)

    def test_invalid_label_raises(self) -> None:
        df = pd.DataFrame({
            "text": ["a"],
            "label": ["phishing"],
        })
        if pytest:
            with pytest.raises(ValueError, match="Invalid label"):
                validate_data(df)
        else:
            try:
                validate_data(df)
                raise AssertionError("Expected ValueError")
            except ValueError as e:
                assert "Invalid label" in str(e)

    def test_drops_na_rows(self) -> None:
        df = pd.DataFrame({
            "text": ["ok", None],
            "label": ["spam", "ham"],
        })
        result = validate_data(df)
        assert len(result) == 1
