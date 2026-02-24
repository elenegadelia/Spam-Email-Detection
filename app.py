"""
Streamlit UI for the Spam Email Detection system.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import traceback
from io import BytesIO
from pathlib import Path

import pandas as pd
import streamlit as st

from src.config.config import get_config
from src.utils.common import find_latest_artifact

# ── Page config ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Spam Email Detector",
    page_icon="📧",
    layout="wide",
)


# ── Sidebar ────────────────────────────────────────────────────────────
def _sidebar() -> None:
    cfg = get_config()
    st.sidebar.title("⚙️ Configuration")
    st.sidebar.markdown("---")

    model_path = find_latest_artifact(cfg.paths.models_dir, "best_model")
    vec_path = find_latest_artifact(cfg.paths.vectorizers_dir, "tfidf_vectorizer")

    if model_path:
        st.sidebar.success(f"**Model:** `{model_path.name}`")
    else:
        st.sidebar.error("No trained model found.")

    if vec_path:
        st.sidebar.success(f"**Vectorizer:** `{vec_path.name}`")
    else:
        st.sidebar.error("No vectorizer found.")

    st.sidebar.markdown("---")
    st.sidebar.caption(
        "If artifacts are missing, run:\n"
        "```\npython main.py train\n```"
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Paths**")
    st.sidebar.text(f"Models:      {cfg.paths.models_dir}")
    st.sidebar.text(f"Vectorizers: {cfg.paths.vectorizers_dir}")
    st.sidebar.text(f"Metrics:     {cfg.paths.metrics_dir}")
    st.sidebar.text(f"Logs:        {cfg.paths.logs_dir}")


# ── Load pipeline (cached) ────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model …")
def _load_pipeline():
    """Import and initialise the prediction pipeline, caching it across
    Streamlit reruns."""
    from src.pipeline.prediction_pipeline import PredictionPipeline

    return PredictionPipeline()


# ── Single email tab ──────────────────────────────────────────────────
def _tab_single() -> None:
    st.subheader("✉️ Classify a Single Email")
    email_text = st.text_area(
        "Paste the email body below:",
        height=220,
        placeholder="Dear user, congratulations! You have won …",
    )

    if st.button("🔍 Predict", type="primary", key="btn_single"):
        if not email_text.strip():
            st.warning("Please enter some email text first.")
            return
        try:
            pipe = _load_pipeline()
            with st.spinner("Classifying …"):
                result = pipe.predict(email_text)
            _show_result(result)
        except FileNotFoundError as e:
            st.error(str(e))
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.code(traceback.format_exc())


def _show_result(result: dict) -> None:
    """Render prediction result with nice formatting."""
    label = result["label"]
    conf = result["confidence"]
    model = result["model"]

    col1, col2, col3 = st.columns(3)
    if label == "spam":
        col1.metric("Prediction", "🚫 SPAM")
    elif label == "ham":
        col1.metric("Prediction", "✅ HAM")
    else:
        col1.metric("Prediction", "❓ UNKNOWN")

    col2.metric("Confidence", f"{conf:.1%}")
    col3.metric("Model", model)


# ── Batch mbox tab ────────────────────────────────────────────────────
def _tab_batch() -> None:
    st.subheader("📦 Batch Classify an MBOX File")
    uploaded = st.file_uploader(
        "Upload a `.mbox` file",
        type=["mbox"],
        help="Standard Unix mbox format.",
    )

    if uploaded is not None:
        if st.button("⚡ Process", type="primary", key="btn_batch"):
            try:
                pipe = _load_pipeline()
                with st.spinner("Parsing & classifying — this may take a moment …"):
                    df = pipe.predict_batch_mbox(uploaded)

                st.success(f"Processed **{len(df)}** emails.")
                st.dataframe(df, use_container_width=True)

                # Download CSV button
                csv_bytes = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Download CSV",
                    data=csv_bytes,
                    file_name="spam_detection_results.csv",
                    mime="text/csv",
                )
            except FileNotFoundError as e:
                st.error(str(e))
            except ValueError as e:
                st.warning(str(e))
            except Exception as e:
                st.error(f"Batch processing failed: {e}")
                st.code(traceback.format_exc())


# ── Main ───────────────────────────────────────────────────────────────
def main() -> None:
    _sidebar()

    st.title("📧 Spam Email Detector")
    st.markdown(
        "A machine-learning system that classifies emails as **Spam** or **Ham**. "
        "Train a model via the CLI, then use this interface to classify emails."
    )

    tab_single, tab_batch = st.tabs(["Single Email", "Batch MBOX"])

    with tab_single:
        _tab_single()

    with tab_batch:
        _tab_batch()


if __name__ == "__main__":
    main()
