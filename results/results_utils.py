"""Utilities for inspecting grid-search results across multiple runs.

These helpers make it easy to scan the ``runs`` directory structure for
completed grid searches, aggregate metrics across random seeds, and pick the
best configuration for a given metric.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Set

import pandas as pd


def _seed_from_path(path: Path) -> Optional[int]:
    """Extract the integer seed from a directory named ``seed_<int>``."""

    try:
        return int(path.name.split("_")[-1])
    except (IndexError, ValueError):
        return None


def _metric_should_maximize(metric: str) -> bool:
    """Infer whether a metric should be maximized or minimized."""

    metric_lower = metric.lower()
    minimize_tokens = ["loss", "error", "mae", "rmse", "mse"]
    return not any(token in metric_lower for token in minimize_tokens)


def _aggregated_columns_for_strategy(strategy: str) -> Set[str]:
    """Columns that should be averaged instead of grouped for a strategy."""

    mapping = {
        "sasla": {"final_set", "bp_examples", "fw_examples", "optimizer_steps"},
        "us": {"final_set", "bp_examples", "fw_examples", "optimizer_steps", "epochs"},
    }
    return mapping.get(strategy, set())


def _parameter_columns_for_strategy(strategy: str) -> Set[str]:
    """Columns that uniquely identify a grid-search configuration."""

    mapping = {
        "passive": {"lr", "weight_decay", "momentum"},
        "sasla": {"sasla_beta", "sasla_tau", "sasla_select_every"},
        "us": {"initial_labeled", "us_k_per_round", "epochs"},
    }
    return mapping.get(strategy, set())


def _collect_seed_results(task_dir: Path) -> List[pd.DataFrame]:
    """Collect per-seed grid search CSV files beneath ``task_dir``."""

    seed_frames: List[pd.DataFrame] = []
    for seed_dir in sorted(task_dir.glob("seed_*")):
        if not seed_dir.is_dir():
            continue
        grid_path = seed_dir / "grid_results.csv"
        if not grid_path.exists():
            continue
        seed = _seed_from_path(seed_dir)
        df = pd.read_csv(grid_path)
        if seed is not None:
            df["seed"] = seed
        seed_frames.append(df)
    return seed_frames


def load_grid_results(
    root_dir: Path,
    strategy: str,
    dataset: str,
    task: str,
) -> pd.DataFrame:
    """Load and concatenate all grid search results for the given settings."""

    task_dir = root_dir / strategy / dataset / f"task_{task}"
    if not task_dir.exists():
        raise FileNotFoundError(f"Task directory not found: {task_dir}")

    seed_frames = _collect_seed_results(task_dir)
    if not seed_frames:
        raise FileNotFoundError(f"No grid_results.csv files found in {task_dir}")

    return pd.concat(seed_frames, ignore_index=True)


def aggregate_grid_results(
    root_dir: Path,
    strategy: str,
    dataset: str,
    task: str,
    metric: str,
    maximize: Optional[bool] = None,
) -> pd.DataFrame:
    """Aggregate grid results across seeds for the chosen metric."""

    full = load_grid_results(root_dir, strategy, dataset, task)
    if metric not in full.columns:
        raise KeyError(f"Metric '{metric}' not found in grid results columns: {list(full.columns)}")

    maximize = _metric_should_maximize(metric) if maximize is None else maximize

    metric_columns = [c for c in full.columns if c.startswith(("val_", "train_", "test_"))]
    aggregate_columns = _aggregated_columns_for_strategy(strategy)
    parameter_columns = _parameter_columns_for_strategy(strategy)
    group_columns = [
        c
        for c in full.columns
        if c not in set(metric_columns) | {"seed"}
    ]
    group_columns = [
        c
        for c in group_columns
        if not (c in aggregate_columns and c not in parameter_columns)
    ]

    agg_kwargs = {
        "num_seeds": ("seed", "nunique"),
        f"{metric}_mean": (metric, "mean"),
        f"{metric}_std": (metric, "std"),
    }
    for col in metric_columns:
        if col == metric:
            continue
        agg_kwargs[f"{col}_mean"] = (col, "mean")
    for col in aggregate_columns:
        if col in full.columns:
            agg_kwargs[f"{col}_mean"] = (col, "mean")
            agg_kwargs[f"{col}_std"] = (col, "std")

    aggregated = (
        full.groupby(group_columns, dropna=False)
        .agg(**agg_kwargs)
        .reset_index()
    )

    mean_column = f"{metric}_mean"
    direction = "desc" if maximize else "asc"
    aggregated = aggregated.sort_values(mean_column, ascending=(direction == "asc"))
    aggregated["maximize_metric"] = maximize
    aggregated["metric_name"] = metric
    return aggregated


def best_grid_configuration(
    root_dir: Path,
    strategy: str,
    dataset: str,
    task: str,
    metric: str,
    maximize: Optional[bool] = None,
) -> pd.Series:
    """Return the aggregated row that achieves the best mean metric."""

    aggregated = aggregate_grid_results(root_dir, strategy, dataset, task, metric, maximize)
    if aggregated.empty:
        raise ValueError("Aggregated grid results are empty.")
    best_row = aggregated.iloc[0]
    return best_row


def summarize_best_configurations(
    root_dir: Path,
    entries: Iterable[tuple[str, str, str]],
    metric: str,
    maximize: Optional[bool] = None,
) -> pd.DataFrame:
    """Convenience helper to batch-evaluate the best config for many settings."""

    records = []
    for strategy, dataset, task in entries:
        best = best_grid_configuration(root_dir, strategy, dataset, task, metric, maximize)
        record = best.to_dict()
        record.update({"strategy": strategy, "dataset": dataset, "task": task})
        records.append(record)
    return pd.DataFrame(records)
