"""Evaluation workflow for the deterministic Orchestrate baseline."""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[2]
CODE_DIR = ROOT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from main import predict_row, read_csv, write_csv  # noqa: E402


EVAL_COLUMNS = [
    "evidence_standard_met",
    "risk_flags",
    "issue_type",
    "object_part",
    "claim_status",
    "valid_image",
    "severity",
]


def load_history(dataset_dir: Path) -> Dict[str, dict]:
    return {row["user_id"]: row for row in read_csv(dataset_dir / "user_history.csv")}


def exact_match(expected: str, actual: str) -> bool:
    return (expected or "").strip().lower() == (actual or "").strip().lower()


def score_rows(expected_rows: List[dict], predicted_rows: List[dict]) -> dict:
    totals = Counter()
    correct = Counter()
    for expected, predicted in zip(expected_rows, predicted_rows):
        for column in EVAL_COLUMNS:
            totals[column] += 1
            if exact_match(expected.get(column, ""), predicted.get(column, "")):
                correct[column] += 1
    per_column = {
        column: (correct[column] / totals[column] if totals[column] else 0.0)
        for column in EVAL_COLUMNS
    }
    macro = sum(per_column.values()) / len(per_column)
    return {"per_column": per_column, "macro": macro}


def predict_sample(strategy: str, dataset_dir: Path) -> List[dict]:
    rows = read_csv(dataset_dir / "sample_claims.csv")
    history = load_history(dataset_dir)
    return [predict_row(row, history, dataset_dir, strategy=strategy) for row in rows]


def markdown_table(scores: Dict[str, dict]) -> str:
    lines = [
        "| strategy | macro exact match | claim_status | issue_type | object_part | severity |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for strategy, score in scores.items():
        cols = score["per_column"]
        lines.append(
            "| {strategy} | {macro:.3f} | {claim_status:.3f} | {issue_type:.3f} | {object_part:.3f} | {severity:.3f} |".format(
                strategy=strategy,
                macro=score["macro"],
                claim_status=cols["claim_status"],
                issue_type=cols["issue_type"],
                object_part=cols["object_part"],
                severity=cols["severity"],
            )
        )
    return "\n".join(lines)


def write_report(report_path: Path, scores: Dict[str, dict], image_count: int) -> None:
    report = f"""# Evaluation Report

## Strategies Compared

{markdown_table(scores)}

`text_only` extracts issue type and object part from the conversation and assumes available evidence supports the claim. `final` adds local image-file validation, prompt-injection risk detection, and user-history risk flags.

## Final Strategy

The submission uses the `final` strategy from `code/main.py`. It is deterministic, dependency-free, reads all required CSVs, checks local image references, and writes the exact output schema.

## Operational Analysis

- Model calls: 0 for sample processing and 0 for test processing.
- Approximate token usage: 0 input and 0 output tokens because no hosted model is called.
- Images processed: {image_count} referenced test images are checked for local availability.
- Approximate cost: $0.00 for the full test set under the current deterministic configuration.
- Runtime: expected to be under a second for the provided CSVs on a typical laptop.
- TPM/RPM considerations: none for the default path. If a VLM is added later, cache per-image observations by file hash and batch claims by object type to avoid repeated image calls.

## Residual Risk

This baseline cannot truly inspect visual damage semantics. It is a stable evaluable floor and a clean integration point for adding a VLM observation layer without changing the output contract.
"""
    report_path.write_text(report, encoding="utf-8")


def main() -> None:
    dataset_dir = ROOT / "dataset"
    expected_rows = read_csv(dataset_dir / "sample_claims.csv")
    strategies = {}
    predictions_by_strategy = {}
    for strategy in ("text_only", "final"):
        predicted = predict_sample(strategy, dataset_dir)
        predictions_by_strategy[strategy] = predicted
        strategies[strategy] = score_rows(expected_rows, predicted)

    out_dir = ROOT / "code" / "evaluation"
    write_csv(out_dir / "sample_predictions_final.csv", predictions_by_strategy["final"])

    test_rows = read_csv(dataset_dir / "claims.csv")
    image_count = sum(len([p for p in row["image_paths"].split(";") if p.strip()]) for row in test_rows)
    write_report(out_dir / "evaluation_report.md", strategies, image_count)

    for strategy, score in strategies.items():
        print(f"{strategy}: macro_exact_match={score['macro']:.3f}")
        for column, value in score["per_column"].items():
            print(f"  {column}: {value:.3f}")


if __name__ == "__main__":
    main()
