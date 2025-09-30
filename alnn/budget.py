
from dataclasses import dataclass
from typing import Optional


class BudgetExceeded(Exception):
    pass


@dataclass
class ComputeTracker:
    compute_budget: int
    score_budget: Optional[int]
    bp_examples: int = 0
    fw_examples: int = 0
    optimizer_steps: int = 0

    def can_backprop(self, batch_size: int) -> bool:
        return self.bp_examples + batch_size <= self.compute_budget

    def record_backprop(self, batch_size: int, optimizer_step: bool = False) -> None:
        if not self.can_backprop(batch_size):
            raise BudgetExceeded("Backprop budget exceeded")
        self.bp_examples += batch_size
        if optimizer_step:
            self.optimizer_steps += 1

    def can_forward(self, count: int) -> bool:
        if self.score_budget is None:
            return True
        return self.fw_examples + count <= self.score_budget

    def record_forward(self, count: int) -> None:
        if not self.can_forward(count):
            raise BudgetExceeded("Forward-only scoring budget exceeded")
        self.fw_examples += count


@dataclass
class EarlyStopping:

    patience: int
    mode: str = "min"
    best: Optional[float] = None
    counter: int = 0
    tol: float = 1e-8

    def step(self, value: float) -> bool:
        if self.best is None:
            self.best = value
            self.counter = 0
            return False
        if self.mode == "min":
            improved = value < self.best - self.tol
        else:
            improved = value > self.best + self.tol
        if improved:
            self.best = value
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience
