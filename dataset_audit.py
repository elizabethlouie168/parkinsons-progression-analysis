"""
Leakage-aware dataset audit tool for biomedical machine learning projects.

This script generates a quick quality-control report for CSV datasets, with emphasis on
issues that can distort model validation in biomedical data: missingness, duplicate rows,
constant columns, highly correlated feature pairs, possible patient/time identifiers, and
potential target leakage columns.

Usage:
    python dataset_audit.py data/parkinsons_updrs.csv --target total_UPDRS --id subject# --out audit_report.md
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd


def _format_percent(value: float) -> str:
    return f"{value:.1%}"


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Could not find CSV file: {path}")
    return pd.read_csv(path)


def summarize_shape(df: pd.DataFrame) -> list[str]:
    return [
        "## Dataset Overview",
        f"- Rows: {df.shape[0]:,}",
        f"- Columns: {df.shape[1]:,}",
        f"- Duplicate rows: {df.duplicated().sum():,}",
        "",
    ]


def summarize_missingness(df: pd.DataFrame, top_n: int = 10) -> list[str]:
    missing = df.isna().mean().sort_values(ascending=False)
    missing = missing[missing > 0].head(top_n)

    lines = ["## Missingness"]
    if missing.empty:
        lines.append("- No missing values detected.")
    else:
        for col, pct in missing.items():
            lines.append(f"- `{col}`: {_format_percent(pct)} missing")
    lines.append("")
    return lines


def summarize_constant_columns(df: pd.DataFrame) -> list[str]:
    nunique = df.nunique(dropna=False)
    constant_cols = nunique[nunique <= 1].index.tolist()

    lines = ["## Constant or Near-Empty Columns"]
    if not constant_cols:
        lines.append("- No constant columns detected.")
    else:
        for col in constant_cols:
            lines.append(f"- `{col}` has only one unique value.")
    lines.append("")
    return lines


def find_correlated_pairs(df: pd.DataFrame, threshold: float = 0.95, top_n: int = 15) -> list[tuple[str, str, float]]:
    numeric = df.select_dtypes(include="number")
    if numeric.shape[1] < 2:
        return []

    corr = numeric.corr().abs()
    pairs: list[tuple[str, str, float]] = []
    cols = corr.columns.tolist()
    for i, col_a in enumerate(cols):
        for col_b in cols[i + 1 :]:
            value = corr.loc[col_a, col_b]
            if pd.notna(value) and value >= threshold:
                pairs.append((col_a, col_b, float(value)))
    return sorted(pairs, key=lambda x: x[2], reverse=True)[:top_n]


def summarize_correlations(df: pd.DataFrame) -> list[str]:
    pairs = find_correlated_pairs(df)
    lines = ["## Highly Correlated Feature Pairs"]
    if not pairs:
        lines.append("- No feature pairs with absolute correlation >= 0.95 detected.")
    else:
        lines.append("These pairs may be redundant or may create leakage-like shortcuts if one column encodes another.")
        for a, b, value in pairs:
            lines.append(f"- `{a}` and `{b}`: r = {value:.3f}")
    lines.append("")
    return lines


def detect_possible_identifier_columns(df: pd.DataFrame) -> list[str]:
    keywords = ["id", "patient", "subject", "visit", "date", "time", "timestamp", "record"]
    possible = []
    for col in df.columns:
        lower = col.lower()
        if any(keyword in lower for keyword in keywords):
            possible.append(col)
    return possible


def summarize_identifiers(df: pd.DataFrame, id_columns: Iterable[str] | None = None) -> list[str]:
    detected = detect_possible_identifier_columns(df)
    provided = list(id_columns or [])
    all_ids = sorted(set(detected + provided))

    lines = ["## Possible Grouping / Identifier Columns"]
    if not all_ids:
        lines.append("- No obvious patient, subject, visit, date, or time columns detected.")
    else:
        lines.append("These columns may require grouped or temporal validation instead of random row splitting.")
        for col in all_ids:
            if col in df.columns:
                lines.append(f"- `{col}`: {df[col].nunique(dropna=True):,} unique values")
            else:
                lines.append(f"- `{col}` was provided but not found in the dataset.")
    lines.append("")
    return lines


def summarize_target_leakage(df: pd.DataFrame, target: str | None) -> list[str]:
    lines = ["## Potential Target Leakage Flags"]
    if not target:
        lines.append("- No target column provided. Re-run with `--target COLUMN_NAME` for leakage screening.")
        lines.append("")
        return lines

    if target not in df.columns:
        lines.append(f"- Target column `{target}` was not found in the dataset.")
        lines.append("")
        return lines

    target_lower = target.lower()
    suspicious_name_matches = [
        col for col in df.columns
        if col != target and (target_lower in col.lower() or col.lower() in target_lower)
    ]

    numeric = df.select_dtypes(include="number")
    suspicious_corrs = []
    if target in numeric.columns:
        target_corr = numeric.corr(numeric_only=True)[target].drop(labels=[target], errors="ignore").abs()
        suspicious_corrs = target_corr[target_corr >= 0.90].sort_values(ascending=False)

    if not suspicious_name_matches and len(suspicious_corrs) == 0:
        lines.append("- No obvious leakage columns detected by name or high target correlation.")
    else:
        if suspicious_name_matches:
            lines.append("Columns with names similar to the target:")
            for col in suspicious_name_matches:
                lines.append(f"- `{col}`")
        if len(suspicious_corrs) > 0:
            lines.append("Columns with absolute correlation >= 0.90 with the target:")
            for col, value in suspicious_corrs.items():
                lines.append(f"- `{col}`: r = {value:.3f}")
    lines.append("")
    return lines


def build_report(df: pd.DataFrame, target: str | None, id_columns: Iterable[str] | None) -> str:
    lines: list[str] = [
        "# Dataset Audit Report",
        "",
        "This report is a quick screening tool for common dataset quality and validation risks in biomedical machine learning.",
        "It does not prove whether a model is valid, but it helps identify issues that should be checked before training.",
        "",
    ]
    lines.extend(summarize_shape(df))
    lines.extend(summarize_missingness(df))
    lines.extend(summarize_constant_columns(df))
    lines.extend(summarize_correlations(df))
    lines.extend(summarize_identifiers(df, id_columns))
    lines.extend(summarize_target_leakage(df, target))
    lines.extend([
        "## Recommended Next Steps",
        "- Use grouped validation when repeated measurements come from the same patient or subject.",
        "- Avoid random row splits if the dataset has longitudinal or repeated-measures structure.",
        "- Remove or justify columns that directly encode the target, visit order, or patient identity.",
        "- Compare naive validation with stricter validation to quantify performance inflation.",
        "",
    ])
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a leakage-aware audit report for a CSV dataset.")
    parser.add_argument("csv_path", type=Path, help="Path to the CSV file to audit.")
    parser.add_argument("--target", type=str, default=None, help="Target column for leakage screening.")
    parser.add_argument("--id", dest="id_columns", action="append", default=[], help="Identifier/group column. Can be used multiple times.")
    parser.add_argument("--out", type=Path, default=Path("audit_report.md"), help="Output Markdown report path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = load_csv(args.csv_path)
    report = build_report(df, target=args.target, id_columns=args.id_columns)
    args.out.write_text(report, encoding="utf-8")
    print(f"Audit report written to {args.out}")


if __name__ == "__main__":
    main()
