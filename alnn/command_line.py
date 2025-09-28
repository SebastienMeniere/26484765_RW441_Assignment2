import argparse
import csv
import math
import random
import time
import warnings
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
from sklearn.model_selection import train_test_split
import torch

from alnn.budget import BudgetExceeded, ComputeTracker, EarlyStopping
from alnn.data import DatasetBundle, load_dataset
from alnn.logging import RunLogger
from alnn.model import make_model, make_optimizer
from alnn.runner import RunnerConfig
from alnn.strategies import passive_training, sasla_strategy, us_classification, us_regression
from alnn.trainer import evaluate_model, get_loss_fn, make_dataloader, train_one_epoch
from torch import nn



# ---------------------------------------------------------------------------
# Determinism helpers
# ---------------------------------------------------------------------------

def set_global_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


def float_list(values: Sequence[str]) -> List[float]:
    return [float(v) for v in values]


def ensure_positive(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be > 0, got {value}")


class _BufferedLogger:
    """In-memory logger used for grid-search trials."""

    def __init__(self) -> None:
        self.curve_rows: List[Dict[str, Any]] = []
        self.subset_rows: List[Dict[str, Any]] = []
        self.labeled_rows: List[Dict[str, Any]] = []

    def record_curve(self, row: Dict[str, Any]) -> None:
        self.curve_rows.append(dict(row))

    def record_subset(self, row: Dict[str, Any]) -> None:
        self.subset_rows.append(dict(row))

    def record_labeled(self, row: Dict[str, Any]) -> None:
        self.labeled_rows.append(dict(row))

    def finalize(self, *args: Any, **kwargs: Any) -> None:
        return None

    def write_grid_results(self, rows: List[Dict[str, Any]]) -> None:
        return None

    def dump_to_directory(self, target_dir: Path) -> None:
        target_dir.mkdir(parents=True, exist_ok=True)
        if self.curve_rows:
            self._write_csv(target_dir / "curve.csv", self.curve_rows)
        if self.subset_rows:
            self._write_csv(target_dir / "subset_sizes.csv", self.subset_rows)
        if self.labeled_rows:
            self._write_csv(target_dir / "labeled_sizes.csv", self.labeled_rows)

    @staticmethod
    def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
        fieldnames = sorted({key for row in rows for key in row.keys()})
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)




# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Experimental framework comparing passive learning, uncertainty sampling, "
            "and SASLA under a unified compute budget."
        )
    )
    parser.add_argument("--task", choices=["classification", "regression"], required=True)
    parser.add_argument("--strategy", choices=["passive", "us", "sasla"], required=True)
    parser.add_argument(
        "--dataset",
        choices=["iris", "banknote", "mnist", "sine1d", "friedman1", "energy"],
        required=True,
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional dataset size cap before splitting.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--momentum", type=float, default=0.0)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--num_hidden_layers", type=int, default=1)
    parser.add_argument(
        "--num_runs",
        type=int,
        default=1,
        help="Number of seeds to evaluate starting from --seed.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=200,
        help="Outer-loop patience (round-level) for uncertainty sampling.",
    )
    parser.add_argument(
        "--us_inner_patience",
        type=int,
        default=200,
        help="Inner training patience (epochs) for uncertainty sampling models.",
    )
    parser.add_argument(
        "--us_outer_patience",
        type=int,
        default=200,
        help="Outer acquisition patience (rounds) for uncertainty sampling.",
    )
    parser.add_argument("--grid_sasla_beta", nargs="+", type=float, default=None)
    parser.add_argument(
        "--grid_sasla_tau",
        nargs="+",
        type=float,
        default=None,
        help="Candidate tau values for SASLA grid search.",
    )
    parser.add_argument(
        "--grid_sasla_select_every",
        nargs="+",
        type=int,
        default=None,
        help="Candidate select-every intervals for SASLA grid search.",
    )
    parser.add_argument("--grid_initial_labeled", nargs="+", type=int, default=None)
    parser.add_argument("--grid_us_k_per_round", nargs="+", type=int, default=None)
    parser.add_argument("--grid_us_epochs", nargs="+", type=int, default=None)
    parser.add_argument("--compute_budget", type=int, required=True)
    parser.add_argument("--score_budget", type=int, default=None)
    parser.add_argument("--warm_start", action="store_true", help="Reuse model weights between AL rounds.")
    parser.add_argument("--sasla_beta", type=float, default=0.2)
    parser.add_argument("--sasla_tau", type=float, default=1.0)
    parser.add_argument("--sasla_select_every", type=int, default=1)
    parser.add_argument(
        "--sasla_sample_sensitivity",
        type=int,
        default=None,
        help="Optionally limit number of samples considered per SASLA pruning step.",
    )
    parser.add_argument("--us_k_per_round", type=int, default=2)
    parser.add_argument("--us_budget", type=int, default=2000)
    parser.add_argument("--us_top_t", type=int, default=512)
    parser.add_argument("--ensemble_size", type=int, default=3)
    parser.add_argument("--initial_labeled", type=int, default=5)
    parser.add_argument(
        "--conflicting_evidence_bonus",
        type=float,
        default=None,
        help="Optional alpha weight for combining margin uncertainty with conflict score in US.",
    )
    parser.add_argument("--grid_search", action="store_true")
    parser.add_argument("--grid_lr", nargs="+", type=float, default=None)
    parser.add_argument("--grid_weight_decay", nargs="+", type=float, default=None)
    parser.add_argument("--grid_momentum", nargs="+", type=float, default=None)
    parser.add_argument("--out_dir", type=str, default="runs")
    parser.add_argument("--pca_dim", type=int, default=None)
    return parser


def parse_args() -> RunnerConfig:
    parser = build_parser()
    args = parser.parse_args()

    if args.num_hidden_layers != 1:
        warnings.warn(
            "Only one hidden layer is supported. Clamping --num_hidden_layers to 1.",
            RuntimeWarning,
        )
        args.num_hidden_layers = 1

    ensure_positive("--batch_size", args.batch_size)
    ensure_positive("--epochs", args.epochs)
    ensure_positive("--patience", args.patience)
    ensure_positive("--compute_budget", args.compute_budget)
    ensure_positive("--us_inner_patience", args.us_inner_patience)
    ensure_positive("--us_outer_patience", args.us_outer_patience)
    ensure_positive("--num_runs", args.num_runs)

    if args.grid_sasla_beta is not None:
        for value in args.grid_sasla_beta:
            if not (0.0 <= value <= 1.0):
                raise ValueError("--grid_sasla_beta values must be between 0 and 1")
    if args.grid_sasla_tau is not None:
        for value in args.grid_sasla_tau:
            if value <= 0.0:
                raise ValueError("--grid_sasla_tau values must be > 0")
    if args.grid_sasla_select_every is not None:
        for value in args.grid_sasla_select_every:
            ensure_positive("--grid_sasla_select_every", value)
    if args.grid_initial_labeled is not None:
        for value in args.grid_initial_labeled:
            ensure_positive("--grid_initial_labeled", value)
    if args.grid_us_k_per_round is not None:
        for value in args.grid_us_k_per_round:
            ensure_positive("--grid_us_k_per_round", value)
    if args.grid_us_epochs is not None:
        for value in args.grid_us_epochs:
            ensure_positive("--grid_us_epochs", value)

    if args.strategy == "sasla":
        if not (0.0 <= args.sasla_beta <= 1.0):
            raise ValueError("--sasla_beta must be between 0 and 1")
        if args.sasla_tau <= 0:
            raise ValueError("--sasla_tau must be > 0")
        ensure_positive("--sasla_select_every", args.sasla_select_every)

    if args.strategy == "us":
        ensure_positive("--us_k_per_round", args.us_k_per_round)
        ensure_positive("--us_top_t", args.us_top_t)
        ensure_positive("--initial_labeled", args.initial_labeled)
        ensure_positive("--us_budget", args.us_budget)
        if args.task == "regression" and args.ensemble_size <= 0:
            raise ValueError("--ensemble_size must be > 0 for regression US")

    if args.grid_search:
        if args.strategy == "passive":
            pass
        elif args.strategy == "sasla":
            if not args.grid_sasla_beta:
                raise ValueError("--grid_sasla_beta must be provided for SASLA grid search")
        elif args.strategy == "us":
            if not (args.grid_initial_labeled and args.grid_us_k_per_round and args.grid_us_epochs):
                raise ValueError(
                    "Uncertainty sampling grid search requires --grid_initial_labeled, --grid_us_k_per_round, and --grid_us_epochs"
                )
        else:
            raise ValueError("Grid search is not supported for this strategy")

    out_dir = Path(args.out_dir).resolve()
    generator = torch.Generator()
    generator.manual_seed(args.seed)

    config = RunnerConfig(
        task=args.task,
        strategy=args.strategy,
        dataset=args.dataset,
        limit=args.limit,
        seed=args.seed,
        lr=args.lr,
        weight_decay=args.weight_decay,
        momentum=args.momentum,
        batch_size=args.batch_size,
        epochs=args.epochs,
        hidden_dim=args.hidden_dim,
        num_hidden_layers=args.num_hidden_layers,
        early_stopping_patience=args.patience,
        us_inner_patience=args.us_inner_patience,
        us_outer_patience=args.us_outer_patience,
        grid_sasla_beta=args.grid_sasla_beta or [args.sasla_beta],
        grid_sasla_tau=args.grid_sasla_tau or [args.sasla_tau],
        grid_sasla_select_every=args.grid_sasla_select_every or [args.sasla_select_every],
        grid_us_initial_labeled=args.grid_initial_labeled or [args.initial_labeled],
        grid_us_k_per_round=args.grid_us_k_per_round or [args.us_k_per_round],
        grid_us_epochs=args.grid_us_epochs or [args.epochs],
        compute_budget=args.compute_budget,
        score_budget=args.score_budget,
        warm_start=args.warm_start,
        sasla_beta=args.sasla_beta,
        sasla_tau=args.sasla_tau,
        sasla_select_every=args.sasla_select_every,
        sasla_sample_sensitivity=args.sasla_sample_sensitivity,
        us_k_per_round=args.us_k_per_round,
        us_budget=args.us_budget,
        us_top_t=args.us_top_t,
        ensemble_size=args.ensemble_size,
        initial_labeled=args.initial_labeled,
        conflicting_evidence_bonus=args.conflicting_evidence_bonus,
        grid_search=args.grid_search,
        grid_lr=args.grid_lr or [args.lr],
        grid_weight_decay=args.grid_weight_decay or [args.weight_decay],
        grid_momentum=args.grid_momentum or [args.momentum],
        out_dir=out_dir,
        pca_dim=args.pca_dim,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        generator=generator,
        num_runs=args.num_runs,
    )
    return config




def passive_grid_search(
    config: RunnerConfig,
    bundle: DatasetBundle,
    base_tracker: ComputeTracker,
    logger: RunLogger,
) -> Tuple[nn.Module, Dict[str, Any]]:
    print("griddying")
    X_train, X_val, y_train, y_val = train_test_split(
        bundle.X_train,
        bundle.y_train,
        test_size=0.2,
        random_state=config.seed,
        stratify=bundle.y_train if config.task == "classification" else None,
    )
    loss_fn = get_loss_fn(config)
    best_score = float("-inf")
    best_params = None
    grid_rows: List[Dict[str, Any]] = []

    for lr in config.grid_lr:
        for wd in config.grid_weight_decay:
            for mom in config.grid_momentum:
                trial_config = RunnerConfig(
                    **{**asdict(config), "lr": lr, "weight_decay": wd, "momentum": mom}
                )
                tracker = ComputeTracker(config.compute_budget, config.score_budget)
                model = make_model(trial_config).to(config.device)
                optimizer = make_optimizer(model, trial_config)
                train_loader = make_dataloader(X_train, y_train, trial_config, shuffle=True)
                trial_stopper = EarlyStopping(config.early_stopping_patience)
                for _ in range(trial_config.epochs):
                    epoch_loss, stopped = train_one_epoch(model, optimizer, loss_fn, train_loader, config.device, tracker)
                    if trial_stopper.step(epoch_loss):
                        break
                    if stopped:
                        break
                val_loader = make_dataloader(X_val, y_val, trial_config, shuffle=False)
                try:
                    _, val_metrics = evaluate_model(model, loss_fn, val_loader, trial_config, tracker)
                except BudgetExceeded:
                    val_metrics = {}
                primary_metric = next(iter(val_metrics.values()), float("nan"))
                grid_rows.append(
                    {
                        "lr": lr,
                        "weight_decay": wd,
                        "momentum": mom,
                        **{f"val_{k}": v for k, v in val_metrics.items()},
                        "bp_examples": tracker.bp_examples,
                        "fw_examples": tracker.fw_examples,
                        "optimizer_steps": tracker.optimizer_steps,
                    }
                )
                if not math.isnan(primary_metric) and primary_metric > best_score:
                    best_score = primary_metric
                    best_params = (lr, wd, mom)

    logger.write_grid_results(grid_rows)

    if best_params is None:
        raise RuntimeError("Grid search failed to evaluate any configuration")

    lr, wd, mom = best_params
    final_config = RunnerConfig(
        **{**asdict(config), "lr": lr, "weight_decay": wd, "momentum": mom}
    )
    tracker = base_tracker
    model = make_model(final_config).to(config.device)
    optimizer = make_optimizer(model, final_config)
    loss_fn = get_loss_fn(final_config)
    train_loader = make_dataloader(bundle.X_train, bundle.y_train, final_config, shuffle=True)
    test_loader = make_dataloader(bundle.X_test, bundle.y_test, final_config, shuffle=False)

    start_time = time.perf_counter()

    final_stopper = EarlyStopping(config.early_stopping_patience)
    for epoch in range(1, final_config.epochs + 1):
        epoch_loss, stopped = train_one_epoch(model, optimizer, loss_fn, train_loader, config.device, tracker)
        train_loader = make_dataloader(bundle.X_train, bundle.y_train, final_config, shuffle=True)
        try:
            train_eval_loader = make_dataloader(bundle.X_train, bundle.y_train, final_config, shuffle=False)
            train_loss, train_metrics = evaluate_model(model, loss_fn, train_eval_loader, final_config, tracker)
            test_loss, test_metrics = evaluate_model(model, loss_fn, test_loader, final_config, tracker)
        except BudgetExceeded:
            break
        should_stop = final_stopper.step(train_loss)
        logger.record_curve(
            {
                "stage": "passive_grid_epoch",
                "iteration": epoch,
                "bp_examples": tracker.bp_examples,
                "fw_examples": tracker.fw_examples,
                "optimizer_steps": tracker.optimizer_steps,
                "train_loss": train_loss,
                **{f"train_{k}": v for k, v in train_metrics.items()},
                "test_loss": test_loss,
                **{f"test_{k}": v for k, v in test_metrics.items()},
            }
        )
        if should_stop or stopped:
            break

    wall_clock = time.perf_counter() - start_time

    train_loader_final = make_dataloader(bundle.X_train, bundle.y_train, final_config, shuffle=False)
    test_loader_final = make_dataloader(bundle.X_test, bundle.y_test, final_config, shuffle=False)
    try:
        train_loss, train_metrics = evaluate_model(model, loss_fn, train_loader_final, final_config, tracker)
        test_loss, test_metrics = evaluate_model(model, loss_fn, test_loader_final, final_config, tracker)
    except BudgetExceeded:
        train_metrics = {}
        test_metrics = {}
        train_loss = float("nan")
        test_loss = float("nan")

    summary = {
        "best_lr": lr,
        "best_weight_decay": wd,
        "best_momentum": mom,
        "train_loss": train_loss,
        **{f"train_{k}": v for k, v in train_metrics.items()},
        "test_loss": test_loss,
        **{f"test_{k}": v for k, v in test_metrics.items()},
        "bp_examples": tracker.bp_examples,
        "fw_examples": tracker.fw_examples,
        "optimizer_steps": tracker.optimizer_steps,
        "num_parameters": sum(p.numel() for p in model.parameters()),
        "wall_clock_seconds": wall_clock,
    }
    return model, summary


def _make_subset_bundle(
    config: RunnerConfig,
    bundle: DatasetBundle,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> DatasetBundle:
    if config.task == "classification" and config.num_classes > 1:
        y_train_cast = y_train.astype(np.int64)
        y_val_cast = y_val.astype(np.int64)
    else:
        y_train_cast = y_train.astype(np.float32)
        y_val_cast = y_val.astype(np.float32)
    return DatasetBundle(
        X_train=X_train.astype(np.float32),
        y_train=y_train_cast,
        X_test=X_val.astype(np.float32),
        y_test=y_val_cast,
        scaler=bundle.scaler,
        meta=bundle.meta,
        pca=bundle.pca,
    )


def sasla_grid_search(
    config: RunnerConfig,
    bundle: DatasetBundle,
    base_tracker: ComputeTracker,
    logger: RunLogger,
) -> Tuple[nn.Module, Dict[str, Any]]:
    if not config.grid_sasla_beta:
        raise ValueError("No values provided for --grid_sasla_beta")

    stratify = bundle.y_train if config.task == "classification" else None
    if stratify is not None:
        stratify = np.asarray(stratify).reshape(-1)
    X_train_sub, X_val_sub, y_train_sub, y_val_sub = train_test_split(
        bundle.X_train,
        bundle.y_train,
        test_size=0.2,
        random_state=config.seed,
        stratify=stratify,
    )

    grid_rows: List[Dict[str, Any]] = []
    best_score = float("-inf") if config.task == "classification" else float("inf")
    best_params: Optional[Tuple[float, float, int]] = None

    for beta in config.grid_sasla_beta:
        for tau in config.grid_sasla_tau:
            for select_every in config.grid_sasla_select_every:
                trial_config = replace(
                    config,
                    sasla_beta=beta,
                    sasla_tau=tau,
                    sasla_select_every=select_every,
                    grid_search=False,
                )
                trial_config.dataset_meta = dict(config.dataset_meta)
                trial_tracker = ComputeTracker(config.compute_budget, config.score_budget)
                trial_bundle = _make_subset_bundle(
                    trial_config,
                    bundle,
                    bundle.X_train,
                    bundle.y_train,
                    bundle.X_test,
                    bundle.y_test,
                )
                trial_logger = _BufferedLogger()
                _, summary = sasla_strategy(trial_config, trial_bundle, trial_tracker, trial_logger)
                test_metrics = {k: v for k, v in summary.items() if k.startswith("test_")}
                val_metric = summary.get("test_accuracy") if config.task == "classification" else summary.get("test_rmse")
                if config.task == "classification":
                    better = val_metric is not None and not math.isnan(val_metric) and val_metric > best_score
                else:
                    better = val_metric is not None and not math.isnan(val_metric) and val_metric < best_score
                grid_row = {
                    "sasla_beta": beta,
                    "sasla_tau": tau,
                    "sasla_select_every": select_every,
                    **test_metrics,
                    "bp_examples": trial_tracker.bp_examples,
                    "fw_examples": trial_tracker.fw_examples,
                    "optimizer_steps": trial_tracker.optimizer_steps,
                    "final_set": summary.get("final_subset_size"),
                }
                grid_rows.append(grid_row)
                trial_dir = (
                    config.run_dir
                    / "grid_trials"
                    / f"beta_{beta:g}_tau_{tau:g}_select_{select_every}"
                )
                trial_logger.dump_to_directory(trial_dir)
                if better:
                    best_score = val_metric
                    best_params = (beta, tau, select_every)

    logger.write_grid_results(grid_rows)

    if best_params is None:
        raise RuntimeError("SASLA grid search did not produce a valid configuration")

    best_beta, best_tau, best_select_every = best_params
    final_config = replace(
        config,
        sasla_beta=best_beta,
        sasla_tau=best_tau,
        sasla_select_every=best_select_every,
        grid_search=False,
    )
    final_config.dataset_meta = dict(config.dataset_meta)
    config.sasla_beta = best_beta
    config.sasla_tau = best_tau
    config.sasla_select_every = best_select_every
    model, summary = sasla_strategy(final_config, bundle, base_tracker, logger)
    summary["best_sasla_beta"] = best_beta
    summary["best_sasla_tau"] = best_tau
    summary["best_sasla_select_every"] = best_select_every
    return model, summary


def us_grid_search(
    config: RunnerConfig,
    bundle: DatasetBundle,
    base_tracker: ComputeTracker,
    logger: RunLogger,
) -> Tuple[nn.Module, Dict[str, Any]]:
    if not (config.grid_us_initial_labeled and config.grid_us_k_per_round and config.grid_us_epochs):
        raise ValueError("Uncertainty sampling grid search requires parameter grids")

    stratify = bundle.y_train if config.task == "classification" else None
    if stratify is not None:
        stratify = np.asarray(stratify).reshape(-1)
    X_train_sub, X_val_sub, y_train_sub, y_val_sub = train_test_split(
        bundle.X_train,
        bundle.y_train,
        test_size=0.2,
        random_state=config.seed,
        stratify=stratify,
    )

    grid_rows: List[Dict[str, Any]] = []
    if config.task == "classification":
        best_score = float("-inf")
    else:
        best_score = float("inf")
    best_params: Optional[Tuple[int, int, int]] = None

    for init in config.grid_us_initial_labeled:
        if init >= len(X_train_sub):
            continue
        if config.task == "classification" and init < config.num_classes:
            continue
        for k_per_round in config.grid_us_k_per_round:
            for epochs in config.grid_us_epochs:
                trial_config = replace(
                    config,
                    initial_labeled=init,
                    us_k_per_round=k_per_round,
                    epochs=epochs,
                    grid_search=False,
                )
                trial_config.dataset_meta = dict(config.dataset_meta)
                trial_tracker = ComputeTracker(config.compute_budget, config.score_budget)
                trial_bundle = _make_subset_bundle(
                    trial_config,
                    bundle,
                    X_train_sub,
                    y_train_sub,
                    X_val_sub,
                    y_val_sub,
                )
                trial_logger = _BufferedLogger()
                if config.task == "classification":
                    _, summary = us_classification(trial_config, trial_bundle, trial_tracker, trial_logger)
                    val_metric = summary.get("test_accuracy")
                else:
                    _, summary = us_regression(trial_config, trial_bundle, trial_tracker, trial_logger)
                    val_metric = summary.get("test_rmse")
                test_metrics = {k: v for k, v in summary.items() if k.startswith("test_")}
                row = {
                    "initial_labeled": init,
                    "us_k_per_round": k_per_round,
                    "epochs": epochs,
                    **test_metrics,
                    "bp_examples": trial_tracker.bp_examples,
                    "fw_examples": trial_tracker.fw_examples,
                    "optimizer_steps": trial_tracker.optimizer_steps,
                    "final_set": summary.get("final_labeled")
                }
                grid_rows.append(row)
                trial_dir = (
                    config.run_dir
                    / "grid_trials"
                    / f"init_{init}_k_{k_per_round}_epochs_{epochs}"
                )
                trial_logger.dump_to_directory(trial_dir)
                if config.task == "classification":
                    better = val_metric is not None and not math.isnan(val_metric) and val_metric > best_score
                else:
                    better = val_metric is not None and not math.isnan(val_metric) and val_metric < best_score
                if better:
                    best_score = val_metric
                    best_params = (init, k_per_round, epochs)

    logger.write_grid_results(grid_rows)

    if best_params is None:
        raise RuntimeError("Uncertainty sampling grid search did not produce a valid configuration")

    init_best, k_best, epochs_best = best_params
    final_config = replace(
        config,
        initial_labeled=init_best,
        us_k_per_round=k_best,
        epochs=epochs_best,
        grid_search=False,
    )
    final_config.dataset_meta = dict(config.dataset_meta)
    config.initial_labeled = init_best
    config.us_k_per_round = k_best
    config.epochs = epochs_best
    if config.task == "classification":
        model, summary = us_classification(final_config, bundle, base_tracker, logger)
    else:
        model, summary = us_regression(final_config, bundle, base_tracker, logger)
    summary["best_initial_labeled"] = init_best
    summary["best_us_k_per_round"] = k_best
    summary["best_epochs"] = epochs_best
    return model, summary


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def make_run_directory(config: RunnerConfig) -> Path:
    run_dir = (
        config.out_dir
        / config.strategy
        / config.dataset
        / f"task_{config.task}"
        / f"seed_{config.seed}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _run_single(config: RunnerConfig) -> Path:
    set_global_seed(config.seed)
    generator = torch.Generator()
    generator.manual_seed(config.seed)
    config.generator = generator

    bundle = load_dataset(config)
    run_dir = make_run_directory(config)
    config.run_dir = run_dir
    logger = RunLogger(run_dir)

    tracker = ComputeTracker(config.compute_budget, config.score_budget)

    if config.grid_search:
        if config.strategy == "passive":
            model, summary = passive_grid_search(config, bundle, tracker, logger)
        elif config.strategy == "sasla":
            model, summary = sasla_grid_search(config, bundle, tracker, logger)
        elif config.strategy == "us":
            model, summary = us_grid_search(config, bundle, tracker, logger)
        else:
            raise ValueError("Grid search not supported for this strategy")
    elif config.strategy == "passive":
        model, summary = passive_training(config, bundle, tracker, logger)
    elif config.strategy == "us" and config.task == "classification":
        model, summary = us_classification(config, bundle, tracker, logger)
    elif config.strategy == "us" and config.task == "regression":
        model, summary = us_regression(config, bundle, tracker, logger)
    elif config.strategy == "sasla":
        model, summary = sasla_strategy(config, bundle, tracker, logger)
    else:
        raise ValueError(f"Unsupported combination: {config.strategy} and {config.task}")

    logger.finalize(config, summary, model, bundle.scaler, bundle.pca)
    return run_dir


def run(config: RunnerConfig) -> None:
    if config.num_runs <= 1:
        _run_single(config)
        return

    base_seed = config.seed
    combined_rows: List[Dict[str, Any]] = []
    run_root = config.out_dir / config.strategy / config.dataset / f"task_{config.task}"

    for offset in range(config.num_runs):
        seed = base_seed + offset
        seeded_config = replace(
            config,
            seed=seed,
            num_runs=1,
            run_dir=Path(),
            dataset_meta={},
        )
        seeded_config.generator = torch.Generator()
        seeded_config.generator.manual_seed(seed)
        run_dir = _run_single(seeded_config)
        curve_path = run_dir / "curve.csv"
        if curve_path.exists():
            with curve_path.open("r", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    combined_rows.append({"seed": seed, **row})

    if combined_rows:
        summary_path = run_root / f"curves_seeds_{base_seed}_to_{base_seed + config.num_runs - 1}.csv"
        _write_csv(summary_path, combined_rows)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    config = parse_args()
    run(config)


if __name__ == "__main__":
    main()
