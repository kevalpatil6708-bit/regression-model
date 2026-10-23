from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from mlproject.config import ARTIFACTS_DIR, DATA_DIR
from mlproject.data import write_datasets
from mlproject.predict import predict_classification, predict_regression
from mlproject.training import train_classification, train_regression


def _train(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir)
    artifacts_dir = Path(args.artifacts_dir)

    regression_path = data_dir / "regression.csv"
    classification_path = data_dir / "classification.csv"
    if args.force or not (regression_path.exists() and classification_path.exists()):
        write_datasets(data_dir, n_samples=args.n_samples, seed=args.seed)
        print(f"Generated datasets in {data_dir}")

    results: dict[str, dict] = {}
    if args.task in {"all", "regression"}:
        data = pd.read_csv(regression_path)
        result = train_regression(data, artifacts_dir)
        results["regression"] = result["metrics"]
        _write_metrics(artifacts_dir / "regression_metrics.json", result["metrics"])
        print(f"Regression model saved to {result['path']}")

    if args.task in {"all", "classification"}:
        data = pd.read_csv(classification_path)
        result = train_classification(data, artifacts_dir)
        results["classification"] = result["metrics"]
        _write_metrics(
            artifacts_dir / "classification_metrics.json", result["metrics"]
        )
        print(f"Classification model saved to {result['path']}")

    print(json.dumps(results, indent=2))


def _write_metrics(path: Path, metrics: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def _predict(args: argparse.Namespace) -> None:
    artifacts_dir = Path(args.artifacts_dir)
    input_path = Path(args.input)
    frame = pd.read_csv(input_path)

    if args.task == "regression":
        output = predict_regression(
            frame, artifacts_dir / "regression_model.joblib"
        )
    else:
        output = predict_classification(
            frame,
            artifacts_dir / "classification_model.joblib",
            threshold=args.threshold,
        )

    if args.output:
        output.to_csv(args.output, index=False)
        print(f"Predictions written to {args.output}")
    else:
        print(output.to_csv(index=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mlproject",
        description="Train and use regression and classification models.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Generate data and train models")
    train_parser.add_argument(
        "--task", choices=["all", "regression", "classification"], default="all"
    )
    train_parser.add_argument("--n-samples", type=int, default=5000)
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    train_parser.add_argument("--artifacts-dir", type=Path, default=ARTIFACTS_DIR)
    train_parser.add_argument("--force", action="store_true")
    train_parser.set_defaults(func=_train)

    predict_parser = subparsers.add_parser("predict", help="Score a CSV file")
    predict_parser.add_argument(
        "--task", choices=["regression", "classification"], required=True
    )
    predict_parser.add_argument("--input", required=True, type=Path)
    predict_parser.add_argument("--output", type=Path)
    predict_parser.add_argument("--threshold", type=float, default=0.5)
    predict_parser.add_argument("--artifacts-dir", type=Path, default=ARTIFACTS_DIR)
    predict_parser.set_defaults(func=_predict)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


__all__ = ["build_parser", "main"]
