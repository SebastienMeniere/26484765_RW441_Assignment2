import math
from typing import Dict
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from torch import nn

from alnn.runner import RunnerConfig


def make_model(config: RunnerConfig) -> nn.Module:
    """Create a shallow network that always returns logits.

    Hidden layers use a sigmoid activation for every task. For classification we keep
    the output linear so downstream losses/metrics can operate on raw logits;
    `classification_probabilities` converts them to probabilities (sigmoid for
    binary, softmax for multi-class). Regression keeps the linear output as well.
    """

    output_dim = 1 if config.task == "regression" or config.num_classes == 1 else config.num_classes
    hidden_activation = nn.Sigmoid()

    model = nn.Sequential(
        nn.Linear(config.input_dim, config.hidden_dim),
        hidden_activation,
        nn.Linear(config.hidden_dim, output_dim),
    )
    return model


def make_optimizer(model: nn.Module, config: RunnerConfig) -> torch.optim.Optimizer:
    return torch.optim.SGD(
        model.parameters(),
        lr=config.lr,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
    )


def classification_probabilities(logits: torch.Tensor, num_classes: int) -> torch.Tensor:
    if num_classes == 1:
        probs = torch.sigmoid(logits)
        probs = torch.cat([1 - probs, probs], dim=1)
        return probs
    return torch.softmax(logits, dim=1)


def compute_classification_metrics(
    y_true: np.ndarray,
    logits: np.ndarray,
    num_classes: int,
) -> Dict[str, float]:
    if not np.isfinite(logits).all() or not np.isfinite(y_true).all():
        return {
            "accuracy": float("nan"),
            "macro_f1": float("nan"),
            "roc_auc": float("nan"),
            "log_loss": float("nan"),
        }
    if num_classes == 1:
        logits = logits.reshape(-1, 1)
        probs = 1.0 / (1.0 + np.exp(-logits))
        preds = (probs >= 0.5).astype(np.int64)
        y_int = y_true.astype(np.int64)
        acc = accuracy_score(y_int, preds)
        f1 = f1_score(y_int, preds, average="macro", zero_division=0)
        try:
            auc = roc_auc_score(y_int, probs)
        except ValueError:
            auc = float("nan")
        try:
            ll = log_loss(y_int, np.concatenate([1 - probs, probs], axis=1))
        except ValueError:
            ll = float("nan")
        return {
            "accuracy": float(acc),
            "macro_f1": float(f1),
            "roc_auc": float(auc),
            "log_loss": float(ll),
        }
    probs = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    preds = probs.argmax(axis=1)
    y_int = y_true.astype(np.int64)
    acc = accuracy_score(y_int, preds)
    f1 = f1_score(y_int, preds, average="macro", zero_division=0)
    try:
        ll = log_loss(y_int, probs)
    except ValueError:
        ll = float("nan")
    return {
        "accuracy": float(acc),
        "macro_f1": float(f1),
        "roc_auc": float("nan"),
        "log_loss": float(ll),
    }


def compute_regression_metrics(y_true: np.ndarray, preds: np.ndarray) -> Dict[str, float]:
    if not np.isfinite(preds).all() or not np.isfinite(y_true).all():
        return {"rmse": float("nan"), "mae": float("nan"), "r2": float("nan")}
    mse = mean_squared_error(y_true, preds)
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(y_true, preds)
    r2 = r2_score(y_true, preds)
    return {"rmse": float(rmse), "mae": float(mae), "r2": float(r2)}


def prepare_targets(y: np.ndarray, config: RunnerConfig) -> torch.Tensor:
    if config.task == "classification" and config.num_classes > 1:
        return torch.from_numpy(y.astype(np.int64))
    if config.task == "classification":
        return torch.from_numpy(y.astype(np.float32)).view(-1, 1)
    return torch.from_numpy(y.astype(np.float32)).view(-1, 1)
