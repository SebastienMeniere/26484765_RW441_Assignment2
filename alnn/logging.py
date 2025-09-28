import csv
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import joblib
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from torch import nn
from sklearn.discriminant_analysis import StandardScaler
from alnn.runner import RunnerConfig

def config_to_json_dict(config: RunnerConfig) -> Dict[str, Any]:
    """Return a JSON-serialisable dictionary for the run configuration."""
    data = asdict(config)
    # Replace objects that JSON cannot serialise directly.
    data["out_dir"] = str(config.out_dir)
    data["run_dir"] = str(config.run_dir)
    data["device"] = str(config.device)
    data.pop("generator", None)
    return data

class RunLogger:
    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.curve_rows: List[Dict[str, Any]] = []
        self.subset_rows: List[Dict[str, Any]] = []
        self.labeled_rows: List[Dict[str, Any]] = []
        self.summary: Dict[str, Any] = {}

    def record_curve(self, row: Dict[str, Any]) -> None:
        self.curve_rows.append(row)

    def record_subset(self, row: Dict[str, Any]) -> None:
        self.subset_rows.append(row)

    def record_labeled(self, row: Dict[str, Any]) -> None:
        self.labeled_rows.append(row)

    def finalize(
        self,
        config: RunnerConfig,
        summary: Dict[str, Any],
        model: nn.Module,
        scaler: StandardScaler,
        pca: Optional[PCA],
    ) -> None:
        self.summary = summary
        self._write_json(self.run_dir / "summary.json", summary)
        if self.curve_rows:
            self._write_csv(self.run_dir / "curve.csv", self.curve_rows)
        if self.subset_rows:
            self._write_csv(self.run_dir / "subset_sizes.csv", self.subset_rows)
        if self.labeled_rows:
            self._write_csv(self.run_dir / "labeled_sizes.csv", self.labeled_rows)
        torch.save({"model_state_dict": model.state_dict()}, self.run_dir / "model.pt")
        joblib.dump(scaler, self.run_dir / "scaler.pkl")
        if pca is not None:
            joblib.dump(pca, self.run_dir / "pca.pkl")
        config_path = self.run_dir / "config.json"
        self._write_json(config_path, config_to_json_dict(config))

    def write_grid_results(self, rows: List[Dict[str, Any]]) -> None:
        if rows:
            self._write_csv(self.run_dir / "grid_results.csv", rows)

    def _write_json(self, path: Path, payload: Dict[str, Any]) -> None:
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def _write_csv(self, path: Path, rows: List[Dict[str, Any]]) -> None:
        fieldnames = sorted({key for row in rows for key in row.keys()})
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
