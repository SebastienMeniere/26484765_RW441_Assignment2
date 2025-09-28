from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import torch

@dataclass
class RunnerConfig:
    task: str
    strategy: str
    dataset: str
    limit: Optional[int]
    seed: int
    lr: float
    weight_decay: float
    momentum: float
    batch_size: int
    epochs: int
    hidden_dim: int
    num_hidden_layers: int
    early_stopping_patience: int
    us_inner_patience: int
    us_outer_patience: int
    grid_sasla_beta: List[float]
    grid_sasla_tau: List[float]
    grid_sasla_select_every: List[int]
    grid_us_initial_labeled: List[int]
    grid_us_k_per_round: List[int]
    grid_us_epochs: List[int]
    compute_budget: int
    score_budget: Optional[int]
    warm_start: bool
    sasla_beta: float
    sasla_tau: float
    sasla_select_every: int
    sasla_sample_sensitivity: Optional[int]
    us_k_per_round: int
    us_budget: int
    us_top_t: int
    ensemble_size: int
    initial_labeled: int
    conflicting_evidence_bonus: Optional[float]
    grid_search: bool
    grid_lr: List[float]
    grid_weight_decay: List[float]
    grid_momentum: List[float]
    out_dir: Path
    pca_dim: Optional[int]
    device: torch.device
    generator: torch.Generator
    num_runs: int = 1
    dataset_meta: Dict[str, Any] = field(default_factory=dict)
    run_dir: Path = Path()

    def is_binary(self) -> bool:
        return self.dataset_meta.get("num_classes", 0) == 2

    @property
    def input_dim(self) -> int:
        return int(self.dataset_meta.get("input_dim", 0))

    @property
    def num_classes(self) -> int:
        return int(self.dataset_meta.get("num_classes", 1))
