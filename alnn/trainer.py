import math
import warnings
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from alnn.budget import BudgetExceeded, ComputeTracker
from alnn.model import compute_classification_metrics, compute_regression_metrics, prepare_targets
from alnn.runner import RunnerConfig


def make_dataloader(
    X: np.ndarray,
    y: np.ndarray,
    config: RunnerConfig,
    shuffle: bool = True,
    batch_size: Optional[int] = None,
) -> DataLoader:
    batch = batch_size or config.batch_size
    features = torch.from_numpy(X.astype(np.float32))
    targets = prepare_targets(y, config)
    dataset = TensorDataset(features, targets)
    loader = DataLoader(
        dataset,
        batch_size=batch,
        shuffle=shuffle,
        generator=config.generator,
    )
    return loader


def get_loss_fn(config: RunnerConfig) -> nn.Module:
    if config.task == "classification" and config.num_classes == 1:
        return nn.BCEWithLogitsLoss()
    if config.task == "classification":
        return nn.CrossEntropyLoss()
    return nn.MSELoss()


def train_one_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    loader: DataLoader,
    device: torch.device,
    tracker: ComputeTracker,
) -> Tuple[float, bool]:
    model.train()
    total_loss = 0.0
    total_items = 0
    stop_early = False
    for xb, yb in loader:
        batch_size = xb.size(0)
        if not tracker.can_backprop(batch_size):
            stop_early = True
            break
        optimizer.zero_grad(set_to_none=True)
        xb = xb.to(device)
        yb = yb.to(device)
        logits = model(xb)
        if isinstance(loss_fn, nn.CrossEntropyLoss):
            loss = loss_fn(logits, yb.long())
        else:
            loss = loss_fn(logits, yb)
        if not torch.isfinite(loss):
            warnings.warn(
                RuntimeWarning,
            )
            stop_early = True
            break
        loss.backward()
        optimizer.step()
        tracker.record_backprop(batch_size, optimizer_step=True)
        total_loss += loss.item() * batch_size
        total_items += batch_size
    avg_loss = total_loss / total_items if total_items > 0 else 0.0
    return avg_loss, stop_early


def evaluate_model(
    model: nn.Module,
    loss_fn: nn.Module,
    loader: DataLoader,
    config: RunnerConfig,
    tracker: ComputeTracker,
) -> Tuple[float, Dict[str, float]]:
    model.eval()
    total_loss = 0.0
    total_items = 0
    logits_list: List[np.ndarray] = []
    targets_list: List[np.ndarray] = []
    with torch.no_grad():
        for xb, yb in loader:
            batch_size = xb.size(0)
            if not tracker.can_forward(batch_size):
                raise BudgetExceeded("Forward budget exhausted during evaluation")
            tracker.record_forward(batch_size)
            xb = xb.to(config.device)
            yb = yb.to(config.device)
            logits = model(xb)
            if isinstance(loss_fn, nn.CrossEntropyLoss):
                loss = loss_fn(logits, yb.long())
            else:
                loss = loss_fn(logits, yb)
            total_loss += loss.item() * batch_size
            total_items += batch_size
            logits_list.append(logits.cpu().numpy())
            targets_list.append(yb.cpu().numpy())
    avg_loss = total_loss / total_items if total_items > 0 else 0.0
    if not math.isfinite(avg_loss):
        avg_loss = float("nan")

    logits_np = np.concatenate(logits_list, axis=0) if logits_list else np.empty((0, 1))
    targets_np = np.concatenate(targets_list, axis=0) if targets_list else np.empty((0, 1))
    if config.task == "classification":
        metrics = compute_classification_metrics(targets_np, logits_np, config.num_classes)
    else:
        metrics = compute_regression_metrics(targets_np, logits_np)
    return avg_loss, metrics

