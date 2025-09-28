import copy
import math
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from alnn.budget import BudgetExceeded, ComputeTracker, EarlyStopping
from alnn.data import DatasetBundle
from alnn.logging import RunLogger
from alnn.model import (
    classification_probabilities,
    compute_regression_metrics,
    make_model,
    make_optimizer,
    prepare_targets,
)
from alnn.runner import RunnerConfig
from alnn.trainer import evaluate_model, get_loss_fn, make_dataloader, train_one_epoch


def _plot_sine1d_predictions(
    model: nn.Module,
    bundle: DatasetBundle,
    config: RunnerConfig,
    output_path: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    model.eval()
    grid_original = np.linspace(-1.0, 1.0, num=512, dtype=np.float32).reshape(-1, 1)
    grid_scaled = bundle.scaler.transform(grid_original)
    grid_tensor = torch.from_numpy(grid_scaled.astype(np.float32)).to(config.device)
    with torch.no_grad():
        preds = model(grid_tensor).cpu().numpy().reshape(-1)

    target_range = config.dataset_meta.get("target_range") or {}
    min_y = float(target_range.get("min", 0.0))
    max_y = float(target_range.get("max", 1.0))
    preds_orig = preds * (max_y - min_y) + min_y

    X_train_scaled = bundle.X_train
    y_train_scaled = bundle.y_train.reshape(-1)
    X_train_orig = bundle.scaler.inverse_transform(X_train_scaled)
    y_train_orig = y_train_scaled * (max_y - min_y) + min_y

    plt.figure(figsize=(8, 4))
    plt.scatter(X_train_orig[:, 0], y_train_orig, s=10, alpha=0.4, label="train")
    plt.plot(grid_original[:, 0], preds_orig, color="red", label="model")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Sine1D Regression Fit")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def passive_training(
    config: RunnerConfig,
    bundle: DatasetBundle,
    tracker: ComputeTracker,
    logger: RunLogger,
) -> Tuple[nn.Module, Dict[str, Any]]:
    model = make_model(config).to(config.device)
    optimizer = make_optimizer(model, config)
    loss_fn = get_loss_fn(config)
    train_loader = make_dataloader(bundle.X_train, bundle.y_train, config, shuffle=True)
    test_loader = make_dataloader(bundle.X_test, bundle.y_test, config, shuffle=False)

    start_time = time.perf_counter()

    early_stopper = EarlyStopping(config.early_stopping_patience)
    budget_exhausted = False

    for epoch in range(1, config.epochs + 1):
        train_loss, stopped = train_one_epoch(model, optimizer, loss_fn, train_loader, config.device, tracker)
        if stopped:
            budget_exhausted = True
        train_loader = make_dataloader(bundle.X_train, bundle.y_train, config, shuffle=True)
        train_eval_loader = make_dataloader(bundle.X_train, bundle.y_train, config, shuffle=False)

        try:
            train_eval_loss, train_metrics = evaluate_model(model, loss_fn, train_eval_loader, config, tracker)
            test_loss, test_metrics = evaluate_model(model, loss_fn, test_loader, config, tracker)
        except BudgetExceeded:
            break

        should_stop = early_stopper.step(train_eval_loss)

        logger.record_curve(
            {
                "stage": "passive_epoch",
                "iteration": epoch,
                "bp_examples": tracker.bp_examples,
                "fw_examples": tracker.fw_examples,
                "optimizer_steps": tracker.optimizer_steps,
                "train_loss": train_eval_loss,
                **{f"train_{k}": v for k, v in train_metrics.items()},
                "test_loss": test_loss,
                **{f"test_{k}": v for k, v in test_metrics.items()},
            }
        )
        if should_stop or budget_exhausted:
            break

    wall_clock = time.perf_counter() - start_time

    train_loader = make_dataloader(bundle.X_train, bundle.y_train, config, shuffle=False)
    test_loader = make_dataloader(bundle.X_test, bundle.y_test, config, shuffle=False)
    try:
        train_loss, train_metrics = evaluate_model(model, loss_fn, train_loader, config, tracker)
        test_loss, test_metrics = evaluate_model(model, loss_fn, test_loader, config, tracker)
    except BudgetExceeded:
        train_metrics = {}
        test_metrics = {}
        train_loss = float("nan")
        test_loss = float("nan")

    summary = {
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
    if config.task == "regression" and config.dataset == "sine1d":
        plot_path = Path(config.run_dir) / "sine_fit.png"
        _plot_sine1d_predictions(model, bundle, config, plot_path)
        summary["sine_plot"] = str(plot_path.name)
    return model, summary


def stratified_initial_indices(y: np.ndarray, n_init: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    unique, counts = np.unique(y, return_counts=True)
    per_class = {cls: max(1, int(round(n_init * (count / len(y))))) for cls, count in zip(unique, counts)}
    chosen = []
    for cls in unique:
        cls_indices = np.where(y == cls)[0]
        take = min(len(cls_indices), per_class[cls])
        chosen.extend(rng.choice(cls_indices, size=take, replace=False).tolist())
    if len(chosen) < n_init:
        remaining = [idx for idx in range(len(y)) if idx not in chosen]
        rng.shuffle(remaining)
        chosen.extend(remaining[: n_init - len(chosen)])
    rng.shuffle(chosen)
    return np.array(chosen[:n_init], dtype=np.int64)


def uniform_initial_indices(n_items: int, n_init: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_items)
    return indices[:n_init]


def us_classification(
    config: RunnerConfig,
    bundle: DatasetBundle,
    tracker: ComputeTracker,
    logger: RunLogger,
) -> Tuple[nn.Module, Dict[str, Any]]:
    X_train = bundle.X_train
    y_train = bundle.y_train.astype(np.int64)
    pool_indices = np.arange(len(X_train))
    if config.initial_labeled < config.num_classes:
        raise ValueError("--initial_labeled must be at least the number of classes")
    labeled = stratified_initial_indices(y_train, config.initial_labeled, config.seed)
    unlabeled = np.array(sorted(set(pool_indices) - set(labeled)))

    loss_fn = get_loss_fn(config)
    test_loader = make_dataloader(bundle.X_test, bundle.y_test, config, shuffle=False)

    remaining_budget = config.us_budget
    model = make_model(config).to(config.device)
    optimizer = make_optimizer(model, config)
    round_id = 0
    start_time = time.perf_counter()

    outer_stopper = EarlyStopping(config.us_outer_patience)

    while remaining_budget > 0 and len(unlabeled) > 0:
        round_id += 1
        labeled_loader = make_dataloader(X_train[labeled], y_train[labeled], config, shuffle=True)
        inner_stopper = EarlyStopping(config.us_inner_patience)
        budget_exhausted = False
        for epoch in range(1, config.epochs + 1):
            epoch_loss, stopped = train_one_epoch(model, optimizer, loss_fn, labeled_loader, config.device, tracker)
            if inner_stopper.step(epoch_loss):
                break
            if stopped:
                budget_exhausted = True
                break
        labeled_loader = make_dataloader(X_train[labeled], y_train[labeled], config, shuffle=False)
        try:
            train_loss, train_metrics = evaluate_model(model, loss_fn, labeled_loader, config, tracker)
            test_loss, test_metrics = evaluate_model(model, loss_fn, test_loader, config, tracker)
        except BudgetExceeded:
            break

        logger.record_curve(
            {
                "stage": "us_round",
                "iteration": round_id,
                "bp_examples": tracker.bp_examples,
                "fw_examples": tracker.fw_examples,
                "optimizer_steps": tracker.optimizer_steps,
                "labeled_size": int(len(labeled)),
                "train_loss": train_loss,
                **{f"train_{k}": v for k, v in train_metrics.items()},
                "test_loss": test_loss,
                **{f"test_{k}": v for k, v in test_metrics.items()},
            }
        )
        logger.record_labeled(
            {
                "round": round_id,
                "labeled": int(len(labeled)),
                "unlabeled": int(len(unlabeled)),
                "bp_examples": tracker.bp_examples,
                "fw_examples": tracker.fw_examples,
            }
        )

        if outer_stopper.step(test_loss):
            break

        if (
            remaining_budget <= 0
            or len(unlabeled) == 0
            or budget_exhausted
            or tracker.bp_examples >= config.compute_budget
        ):
            break

        margin_scores: List[Tuple[float, int, float]] = []
        batch_size = max(1, config.batch_size)
        for start in range(0, len(unlabeled), batch_size):
            batch_indices = unlabeled[start : start + batch_size]
            xb = torch.from_numpy(X_train[batch_indices].astype(np.float32)).to(config.device)
            if not tracker.can_forward(len(batch_indices)):
                remaining_budget = 0
                break
            tracker.record_forward(len(batch_indices))
            logits = model(xb)
            probs = classification_probabilities(logits, config.num_classes)
            if config.num_classes == 1:
                margin = torch.abs(probs[:, 1] - 0.5)
                conflict = probs[:, 0] * probs[:, 1]
            else:
                sorted_probs, _ = torch.sort(probs, dim=1, descending=True)
                margin = sorted_probs[:, 0] - sorted_probs[:, 1]
                conflict = sorted_probs[:, 0] * sorted_probs[:, 1]
            for local_idx, idx in enumerate(batch_indices):
                score_margin = float(margin[local_idx].item())
                conflict_score = float(conflict[local_idx].item())
                if config.conflicting_evidence_bonus is not None:
                    alpha = config.conflicting_evidence_bonus
                    combined = alpha * (-score_margin) + (1.0 - alpha) * conflict_score
                else:
                    combined = -score_margin
                margin_scores.append((combined, idx, score_margin))
        if not margin_scores or remaining_budget <= 0 or tracker.bp_examples >= config.compute_budget:
            break
        margin_scores.sort(key=lambda item: item[0], reverse=True)
        shortlist = margin_scores[: min(config.us_top_t, len(margin_scores))]
        shortlist.sort(key=lambda item: item[2])
        acquire = [idx for _, idx, _ in shortlist[: min(config.us_k_per_round, remaining_budget, len(shortlist))]]
        if not acquire:
            break
        labeled = np.concatenate([labeled, np.array(acquire, dtype=np.int64)])
        unlabeled = np.array([idx for idx in unlabeled if idx not in acquire])
        remaining_budget -= len(acquire)

        if not config.warm_start:
            model = make_model(config).to(config.device)
            optimizer = make_optimizer(model, config)

    wall_clock = time.perf_counter() - start_time

    labeled_loader = make_dataloader(X_train[labeled], y_train[labeled], config, shuffle=False)
    try:
        train_loss, train_metrics = evaluate_model(model, loss_fn, labeled_loader, config, tracker)
        test_loss, test_metrics = evaluate_model(model, loss_fn, test_loader, config, tracker)
    except BudgetExceeded:
        train_metrics = {}
        test_metrics = {}
        train_loss = float("nan")
        test_loss = float("nan")

    summary = {
        "train_loss": train_loss,
        **{f"train_{k}": v for k, v in train_metrics.items()},
        "test_loss": test_loss,
        **{f"test_{k}": v for k, v in test_metrics.items()},
        "bp_examples": tracker.bp_examples,
        "fw_examples": tracker.fw_examples,
        "optimizer_steps": tracker.optimizer_steps,
        "num_parameters": sum(p.numel() for p in model.parameters()),
        "wall_clock_seconds": wall_clock,
        "final_labeled": int(len(labeled)),
    }
    return model, summary


def us_regression(
    config: RunnerConfig,
    bundle: DatasetBundle,
    tracker: ComputeTracker,
    logger: RunLogger,
) -> Tuple[nn.Module, Dict[str, Any]]:
    X_train = bundle.X_train
    y_train = bundle.y_train
    pool_indices = np.arange(len(X_train))
    min_init = min(5, config.batch_size)
    if config.initial_labeled < min_init:
        raise ValueError(f"--initial_labeled must be at least {min_init} for regression")
    member_labeled: List[np.ndarray] = []
    for member_idx in range(config.ensemble_size):
        rng = np.random.default_rng(config.seed + member_idx)
        member_indices = rng.choice(pool_indices, size=config.initial_labeled, replace=False)
        member_labeled.append(np.sort(member_indices.astype(np.int64)))

    shared_labeled = (
        np.unique(np.concatenate(member_labeled)) if member_labeled else np.empty(0, dtype=np.int64)
    )
    unlabeled_mask = np.ones(len(X_train), dtype=bool)
    unlabeled_mask[shared_labeled] = False
    unlabeled = np.nonzero(unlabeled_mask)[0]
    test_loader = make_dataloader(bundle.X_test, bundle.y_test, config, shuffle=False)
    remaining_budget = config.us_budget
    round_id = 0
    start_time = time.perf_counter()

    ensemble: List[nn.Module] = [make_model(config).to(config.device) for _ in range(config.ensemble_size)]
    optimizers = [make_optimizer(member, config) for member in ensemble]
    ensemble_states: List[Dict[str, torch.Tensor]] = [copy.deepcopy(member.state_dict()) for member in ensemble]
    loss_fn = get_loss_fn(config)

    outer_stopper = EarlyStopping(config.us_outer_patience)

    while remaining_budget > 0 and len(unlabeled) > 0:
        round_id += 1
        budget_exhausted = False
        if config.warm_start:
            for idx, member in enumerate(ensemble):
                member.load_state_dict(copy.deepcopy(ensemble_states[idx]))
        for idx, member in enumerate(ensemble):
            member_indices = member_labeled[idx]
            loader = make_dataloader(X_train[member_indices], y_train[member_indices], config, shuffle=True)
            opt = optimizers[idx]
            member_stopper = EarlyStopping(config.us_inner_patience)
            for _ in range(config.epochs):
                epoch_loss, stopped = train_one_epoch(member, opt, loss_fn, loader, config.device, tracker)
                if member_stopper.step(epoch_loss):
                    break
                if stopped:
                    budget_exhausted = True
                    break
            optimizers[idx] = opt
            if budget_exhausted:
                break

        try:
            shared_sorted = np.sort(shared_labeled)
            labeled_loader = make_dataloader(X_train[shared_sorted], y_train[shared_sorted], config, shuffle=False)
            train_loss, train_metrics = _evaluate_regression_ensemble(ensemble, loss_fn, labeled_loader, config, tracker)
            test_loss, test_metrics = _evaluate_regression_ensemble(ensemble, loss_fn, test_loader, config, tracker)
        except BudgetExceeded:
            break

        logger.record_curve(
            {
                "stage": "us_regression_round",
                "iteration": round_id,
                "bp_examples": tracker.bp_examples,
                "fw_examples": tracker.fw_examples,
                "optimizer_steps": tracker.optimizer_steps,
                "labeled_size": int(len(shared_labeled)),
                "train_loss": train_loss,
                **{f"train_{k}": v for k, v in train_metrics.items()},
                "test_loss": test_loss,
                **{f"test_{k}": v for k, v in test_metrics.items()},
            }
        )
        logger.record_labeled(
            {
                "round": round_id,
                "labeled": int(len(shared_labeled)),
                "unlabeled": int(len(unlabeled)),
                "bp_examples": tracker.bp_examples,
                "fw_examples": tracker.fw_examples,
            }
        )

        if outer_stopper.step(test_loss):
            break

        if (
            remaining_budget <= 0
            or len(unlabeled) == 0
            or budget_exhausted
            or tracker.bp_examples >= config.compute_budget
        ):
            break

        ensemble_states = [copy.deepcopy(member.state_dict()) for member in ensemble]

        variances: List[Tuple[float, int]] = []
        batch_size = max(1, config.batch_size)
        for start in range(0, len(unlabeled), batch_size):
            batch_indices = unlabeled[start : start + batch_size]
            xb = torch.from_numpy(X_train[batch_indices].astype(np.float32)).to(config.device)
            preds = []
            for member in ensemble:
                if not tracker.can_forward(len(batch_indices)):
                    remaining_budget = 0
                    break
                tracker.record_forward(len(batch_indices))
                preds.append(member(xb).detach().cpu().numpy())
            if remaining_budget <= 0:
                break
            preds_stack = np.stack(preds, axis=0)
            variance = preds_stack.var(axis=0).reshape(-1)
            for local_idx, idx in enumerate(batch_indices):
                variances.append((float(variance[local_idx]), idx))
        if not variances or remaining_budget <= 0:
            break
        variances.sort(key=lambda item: item[0], reverse=True)
        acquire = [idx for _, idx in variances[: min(config.us_k_per_round, remaining_budget, len(variances))]]
        if not acquire:
            break
        acquire_arr = np.array(acquire, dtype=np.int64)
        if acquire_arr.size > 0:
            for idx in range(config.ensemble_size):
                member_labeled[idx] = np.unique(np.concatenate([member_labeled[idx], acquire_arr]))
            shared_labeled = np.unique(np.concatenate(member_labeled))
            unlabeled_mask = np.ones(len(X_train), dtype=bool)
            unlabeled_mask[shared_labeled] = False
            unlabeled = np.nonzero(unlabeled_mask)[0]
        remaining_budget -= len(acquire)

        if not config.warm_start:
            ensemble = [make_model(config).to(config.device) for _ in range(config.ensemble_size)]
            optimizers = [make_optimizer(member, config) for member in ensemble]
            ensemble_states = [copy.deepcopy(member.state_dict()) for member in ensemble]

    wall_clock = time.perf_counter() - start_time

    reference_model = ensemble[0]
    shared_sorted = np.sort(shared_labeled)
    labeled_loader = make_dataloader(X_train[shared_sorted], y_train[shared_sorted], config, shuffle=False)
    try:
        train_loss, train_metrics = _evaluate_regression_ensemble(ensemble, loss_fn, labeled_loader, config, tracker)
        test_loss, test_metrics = _evaluate_regression_ensemble(ensemble, loss_fn, test_loader, config, tracker)
    except BudgetExceeded:
        train_metrics = {}
        test_metrics = {}
        train_loss = float("nan")
        test_loss = float("nan")

    summary = {
        "train_loss": train_loss,
        **{f"train_{k}": v for k, v in train_metrics.items()},
        "test_loss": test_loss,
        **{f"test_{k}": v for k, v in test_metrics.items()},
        "bp_examples": tracker.bp_examples,
        "fw_examples": tracker.fw_examples,
        "optimizer_steps": tracker.optimizer_steps,
        "num_parameters": sum(p.numel() for p in reference_model.parameters()),
        "wall_clock_seconds": wall_clock,
        "final_labeled": int(len(shared_labeled)),
    }
    return reference_model, summary



def compute_sensitivities(
    model: nn.Module,
    loss_fn: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    indices: Sequence[int],
    config: RunnerConfig,
    tracker: ComputeTracker,
    max_samples: Optional[int] = None,
) -> Dict[int, float]:
    """Compute per-sample sensitivities while respecting shared budgets."""

    model.eval()
    device = config.device
    param_count = sum(p.numel() for p in model.parameters()) or 1

    chosen = list(indices)
    if max_samples is not None and max_samples < len(chosen):
        rng = random.Random(getattr(config, "seed", 0))
        chosen = rng.sample(chosen, max_samples)

    mode = getattr(config, "sasla_sensitivity_mode", "auto").lower()
    input_norm = getattr(config, "sasla_input_norm", "l1").lower()

    def is_classification_loss(fn: nn.Module) -> bool:
        return isinstance(fn, (nn.CrossEntropyLoss, nn.BCEWithLogitsLoss))

    # approx the paper's calculation of the cost -- i know this isnt right
    # x0 = X[0 : 1].to(device).detach().requires_grad_(True)
    # out = model(x0)
    # K = out.shape[1]

    # if not tracker.can_backprop(len(X) // config.batch_size * K) :
    #     return None
    # tracker.record_backprop(len(X) // config.batch_size * K, optimizer_step=False)
    
    use_paper = (mode == "paper") or (mode == "auto" and is_classification_loss(loss_fn))
    use_loss_grad = (mode == "loss_grad") or (mode == "auto" and not is_classification_loss(loss_fn))

    sensitivities: Dict[int, float] = {}

    for idx in chosen:
        if use_paper:
            outputs_per_sample = config.num_classes if config.num_classes > 1 else 1
            if not tracker.can_forward(1) or not tracker.can_backprop(outputs_per_sample):
                break
            xi = X[idx : idx + 1].to(device).detach().requires_grad_(True)
            tracker.record_forward(1)
            with torch.enable_grad():
                out = model(xi)
                if out.ndim == 1:
                    out = out.unsqueeze(1)
                K = out.shape[1]
                
                if not tracker.can_backprop(1):
                    break
                
                if K > 1:
                    probs = torch.softmax(out, dim=1)
                else:
                    probs = torch.sigmoid(out)
                per_out_vals = []
                for k in range(K):
                    grad = torch.autograd.grad(
                        probs[:, k].sum(),
                        xi,
                        retain_graph=(k < K - 1),
                        allow_unused=False,
                    )[0]
                    if input_norm == "l2":
                        val = torch.sqrt((grad * grad).sum(dim=1)) / math.sqrt(grad.shape[1])
                    else:
                        val = grad.abs().sum(dim=1) / grad.shape[1]
                    per_out_vals.append(val.item())
                tracker.record_backprop(1, optimizer_step=False)
                model.zero_grad(set_to_none=True)
                sensitivities[idx] = float(max(per_out_vals))
                continue

        if not use_loss_grad:
            continue
        
        xi = X[idx : idx + 1].to(device)
        yi = y[idx : idx + 1].to(device)
        model.zero_grad(set_to_none=True)
        tracker.record_forward(1)
        logits = model(xi)
        if isinstance(loss_fn, nn.CrossEntropyLoss):
            loss = loss_fn(logits, yi.long())
        else:
            loss = loss_fn(logits, yi)
        if not tracker.can_backprop(1):
                break
        tracker.record_backprop(1, optimizer_step=False)
        loss.backward()
        sq_norm = 0.0
        for param in model.parameters():
            if param.grad is not None:
                sq_norm += param.grad.pow(2).sum().item()
        sensitivities[idx] = math.sqrt(sq_norm / param_count)

    return sensitivities


def _evaluate_regression_ensemble(
    ensemble: Sequence[nn.Module],
    loss_fn: nn.Module,
    loader: torch.utils.data.DataLoader,
    config: RunnerConfig,
    tracker: ComputeTracker,
) -> Tuple[float, Dict[str, float]]:
    for member in ensemble:
        member.eval()

    total_loss = 0.0
    total_items = 0
    preds_accum: List[np.ndarray] = []
    targets_accum: List[np.ndarray] = []

    with torch.no_grad():
        for xb, yb in loader:
            batch_size = xb.size(0)
            xb = xb.to(config.device)
            yb = yb.to(config.device)

            member_outputs: List[torch.Tensor] = []
            for member in ensemble:
                if not tracker.can_forward(batch_size):
                    raise BudgetExceeded("Forward budget exhausted during ensemble evaluation")
                tracker.record_forward(batch_size)
                member_outputs.append(member(xb))

            outputs_stack = torch.stack(member_outputs, dim=0)
            mean_output = outputs_stack.mean(dim=0)

            loss = loss_fn(mean_output, yb)
            total_loss += loss.item() * batch_size
            total_items += batch_size

            preds_accum.append(mean_output.cpu().numpy())
            targets_accum.append(yb.cpu().numpy())

    avg_loss = total_loss / total_items if total_items > 0 else 0.0
    preds_np = np.concatenate(preds_accum, axis=0) if preds_accum else np.empty((0, 1))
    targets_np = np.concatenate(targets_accum, axis=0) if targets_accum else np.empty((0, 1))
    metrics = compute_regression_metrics(targets_np, preds_np)
    return avg_loss, metrics


def sasla_strategy(
    config: RunnerConfig,
    bundle: DatasetBundle,
    tracker: ComputeTracker,
    logger: RunLogger,
) -> Tuple[nn.Module, Dict[str, Any]]:
    X_train = torch.from_numpy(bundle.X_train.astype(np.float32))
    y_train = prepare_targets(bundle.y_train, config)
    loss_fn = get_loss_fn(config)
    model = make_model(config).to(config.device)
    optimizer = make_optimizer(model, config)
    test_loader = make_dataloader(bundle.X_test, bundle.y_test, config, shuffle=False)

    start_time = time.perf_counter()

    early_stopper = EarlyStopping(config.early_stopping_patience)
    budget_exhausted = False

    all_idx = list(range(len(X_train)))
    indices = all_idx.copy()

    for epoch in range(1, config.epochs + 1):
        subset_X = X_train[indices].numpy()
        subset_y = bundle.y_train[indices]
        train_loader = make_dataloader(subset_X, subset_y, config, shuffle=True)

        epoch_loss, stopped = train_one_epoch(
            model, optimizer, loss_fn, train_loader, config.device, tracker
        )
        should_stop = early_stopper.step(epoch_loss)
        if stopped:
            budget_exhausted = True

        selection_triggered = False
        if epoch % config.sasla_select_every == 0 and not should_stop:
            pool = all_idx  # not 'indices'
            sensitivities = compute_sensitivities(
                model, loss_fn, X_train, y_train, pool, config, tracker,
                max_samples=config.sasla_sample_sensitivity
            )
            if sensitivities:
                vals = np.fromiter(sensitivities.values(), dtype=np.float32)
                mean_phi = float(vals.mean())
                beta = getattr(config, "sasla_beta", 0.9)  # paper’s α
                # threshold = config.sasla_tau * median_val
                eps = 1e-12
                if beta >= 1.0 - eps:
                    indices = all_idx.copy()  
                else:
                    threshold = (1.0 - beta) * mean_phi   # paper: τ = (1-α) * mean
                    indices = [i for i, v in sensitivities.items() if v > threshold]
                    # print(len(indices))
                    # print(threshold)
                    
        
        train_loader = make_dataloader(bundle.X_train, bundle.y_train, config, shuffle=True)
        train_eval_loader = make_dataloader(bundle.X_train, bundle.y_train, config, shuffle=False)

        try:
            train_eval_loss, train_metrics = evaluate_model(model, loss_fn, train_eval_loader, config, tracker)
            test_loss, test_metrics = evaluate_model(model, loss_fn, test_loader, config, tracker)
        except BudgetExceeded:
            break
        
        logger.record_curve(
            {
                "stage": "sasla_epoch",
                "iteration": epoch,
                "subset_size": int(len(indices)),
                "train_subset_loss": float(epoch_loss),
                "selection_triggered": bool(selection_triggered),
                "bp_examples": tracker.bp_examples,
                "fw_examples": tracker.fw_examples,
                "optimizer_steps": tracker.optimizer_steps,
                 **{f"train_{k}": v for k, v in train_metrics.items()},
                "test_loss": test_loss,
                **{f"test_{k}": v for k, v in test_metrics.items()},
            }
        )
        if selection_triggered:
            logger.record_subset(
                {
                    "iteration": epoch,
                    "subset_size": int(len(indices)),
                    "sasla_beta": float(getattr(config, "sasla_beta", 0.9)),
                    "sasla_tau": float(getattr(config, "sasla_tau", 1.0)),
                }
            )

        if should_stop or budget_exhausted:
            break

    wall_clock = time.perf_counter() - start_time

    train_loader_final = make_dataloader(bundle.X_train, bundle.y_train, config, shuffle=False)
    test_loader_final = make_dataloader(bundle.X_test, bundle.y_test, config, shuffle=False)
    try:
        train_loss, train_metrics = evaluate_model(model, loss_fn, train_loader_final, config, tracker)
        test_loss, test_metrics = evaluate_model(model, loss_fn, test_loader_final, config, tracker)
    except BudgetExceeded:
        train_metrics = {}
        test_metrics = {}
        train_loss = float("nan")
        test_loss = float("nan")

    summary = {
        "train_loss": train_loss,
        **{f"train_{k}": v for k, v in train_metrics.items()},
        "test_loss": test_loss,
        **{f"test_{k}": v for k, v in test_metrics.items()},
        "bp_examples": tracker.bp_examples,
        "fw_examples": tracker.fw_examples,
        "optimizer_steps": tracker.optimizer_steps,
        "num_parameters": sum(p.numel() for p in model.parameters()),
        "wall_clock_seconds": wall_clock,
        "final_subset_size": len(indices),
    }

    return model, summary
