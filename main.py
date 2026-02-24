#!/usr/bin/env python3
"""
CLI entry-point for the Spam Email Detection system.

Usage:
    python main.py train [path/to/dataset.csv]
    python main.py predict "Your email text here"
    python main.py batch  path/to/file.mbox
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _train(args: argparse.Namespace) -> None:
    from src.pipeline.training_pipeline import run_training_pipeline

    csv = Path(args.csv) if args.csv else None
    summary = run_training_pipeline(csv)
    print("\n✅ Training complete!")
    print(f"   Champion model : {summary['champion']}")
    print(f"   Train size     : {summary['train_size']}")
    print(f"   Test size      : {summary['test_size']}")
    print(f"   Vocab size     : {summary['vocab_size']}")


def _predict(args: argparse.Namespace) -> None:
    from src.pipeline.prediction_pipeline import PredictionPipeline

    pipe = PredictionPipeline()
    result = pipe.predict(args.text)
    print(f"\n📧 Prediction: {result['label'].upper()}")
    print(f"   Confidence : {result['confidence']:.2%}")
    print(f"   Model      : {result['model']}")


def _batch(args: argparse.Namespace) -> None:
    from src.pipeline.prediction_pipeline import PredictionPipeline

    pipe = PredictionPipeline()
    df = pipe.predict_batch_mbox(Path(args.mbox))
    out = Path(args.output)
    df.to_csv(out, index=False)
    print(f"\n✅ Batch results saved to {out}  ({len(df)} emails)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Spam Email Detection — CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # train
    p_train = sub.add_parser("train", help="Run the training pipeline")
    p_train.add_argument("csv", nargs="?", default=None, help="Path to dataset CSV")

    # predict
    p_pred = sub.add_parser("predict", help="Classify a single email text")
    p_pred.add_argument("text", help="Email body text")

    # batch
    p_batch = sub.add_parser("batch", help="Batch-classify an mbox file")
    p_batch.add_argument("mbox", help="Path to .mbox file")
    p_batch.add_argument(
        "-o", "--output", default="batch_results.csv", help="Output CSV path"
    )

    args = parser.parse_args()
    {"train": _train, "predict": _predict, "batch": _batch}[args.command](args)


if __name__ == "__main__":
    main()
