from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
import numpy as np
from sklearn.datasets import fetch_openml, load_iris, make_friedman1
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from alnn.runner import RunnerConfig
@dataclass
class DatasetBundle:
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    scaler: StandardScaler
    meta: Dict[str, Any]
    pca: Optional[PCA] = None

def subsample_limit(X: np.ndarray, y: np.ndarray, limit: Optional[int], seed: int) -> Tuple[np.ndarray, np.ndarray]:
    if limit is None or limit >= len(X):
        return X, y
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(X))[:limit]
    return X[indices], y[indices]

def load_classification_dataset(name: str, limit: Optional[int], seed: int, pca_dim: Optional[int]) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    if name == "iris":
        data = load_iris()
        X = data.data.astype(np.float32)
        y_raw = data.target
        target_names = list(data.target_names)
    elif name == "banknote":
        dataset = fetch_openml("banknote-authentication", version=1, as_frame=False)
        X = dataset.data.astype(np.float32)
        y_raw = dataset.target
        target_names = None
    elif name == "mnist":
        dataset = fetch_openml("mnist_784", version=1, as_frame=False)
        X = dataset.data.astype(np.float32) / 255.0
        y_raw = dataset.target
        target_names = None
    else:
        raise ValueError(f"Unsupported classification dataset: {name}")

    y_raw = np.asarray(y_raw)
    if y_raw.dtype.kind not in "iu":
        y_raw = y_raw.astype(str)
    unique_classes, inverse = np.unique(y_raw, return_inverse=True)
    y = inverse.astype(np.int64)

    if target_names is None or len(target_names) != len(unique_classes):
        target_names = [str(cls) for cls in unique_classes]

    X, y = subsample_limit(X, y, limit, seed)

    if pca_dim is not None and pca_dim < X.shape[1]:
        pca = PCA(n_components=pca_dim, random_state=seed)
        X = pca.fit_transform(X)
        meta = {"pca": pca, "class_labels": [str(cls) for cls in unique_classes]}
    else:
        pca = None
        meta = {"class_labels": [str(cls) for cls in unique_classes]}

    num_classes = len(unique_classes)
    return X, y, {
        "num_classes": num_classes,
        "target_names": target_names,
        "pca": pca,
        "input_dim": X.shape[1],
        "class_labels": [str(cls) for cls in unique_classes],
    }


def generate_regression_dataset(name: str, limit: Optional[int], seed: int) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(seed)
    if name == "sine1d":
        n_samples = limit or 5000
        X = rng.uniform(-1.0, 1.0, size=(n_samples, 1)).astype(np.float32)
        noise = rng.normal(loc=0.0, scale=0.05, size=(n_samples,)).astype(np.float32)
        y = (np.sin(2 * np.pi * X[:, 0]) + noise).astype(np.float32)
        y = y.reshape(-1, 1)
        # plt.scatter(X, y)
        # plt.show()
        return X, y, {"input_dim": X.shape[1], "num_classes": 1}
    if name == "friedman1":
        n_samples = limit or 5000
        X, y = make_friedman1(n_samples=n_samples, noise=1.0, random_state=seed)
        X = X.astype(np.float32)
        y = y.astype(np.float32).reshape(-1, 1)
        return X, y, {"input_dim": X.shape[1], "num_classes": 1}
    if name == "energy":
        dataset = fetch_openml(data_id=1472, as_frame=False)
        X = dataset.data.astype(np.float32)
        y_raw = np.asarray(dataset.target)
        if y_raw.ndim == 1:
            y = y_raw.astype(np.float32).reshape(-1, 1)
        else:
            y = y_raw.astype(np.float32)[:, 0].reshape(-1, 1)
        X, y = subsample_limit(X, y, limit, seed)
        return X, y, {"input_dim": X.shape[1], "num_classes": 1}
    raise ValueError(f"Unsupported regression dataset: {name}")


def load_dataset(config: RunnerConfig) -> DatasetBundle:
    if config.task == "classification" and config.dataset in {"sine1d", "friedman1", "energy"}:
        raise ValueError("Regression datasets cannot be used with classification task")
    if config.task == "regression" and config.dataset in {"iris", "banknote", "mnist"}:
        raise ValueError("Classification datasets cannot be used with regression task")

    if config.task == "classification":
        X, y, meta = load_classification_dataset(config.dataset, config.limit, config.seed, config.pca_dim)
        stratify = y
    else:
        X, y, meta = generate_regression_dataset(config.dataset, config.limit, config.seed)
        stratify = None
    print(len(X))
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=config.seed,
        stratify=stratify,
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    target_range: Optional[Dict[str, float]] = None
    if config.task == "regression":
        y_train_min = float(np.min(y_train))
        y_train_max = float(np.max(y_train))
        denom = y_train_max - y_train_min
        if denom == 0.0:
            denom = 1.0
        y_train = (y_train - y_train_min) / denom
        y_test = (y_test - y_train_min) / denom
        target_range = {"min": y_train_min, "max": y_train_max}

    config.dataset_meta = {
        "input_dim": X_train.shape[1],
        "num_classes": int(meta.get("num_classes", 1)),
        "target_names": meta.get("target_names"),
        "pca_dim": config.pca_dim,
        "original_train_size": int(len(X_train)),
        "target_range": target_range,
    }

    bundle = DatasetBundle(
        X_train=X_train.astype(np.float32),
        y_train=y_train.astype(np.float32 if config.task == "regression" or config.dataset_meta["num_classes"] == 1 else np.int64),
        X_test=X_test.astype(np.float32),
        y_test=y_test.astype(np.float32 if config.task == "regression" or config.dataset_meta["num_classes"] == 1 else np.int64),
        scaler=scaler,
        meta=meta,
        pca=meta.get("pca"),
    )
    return bundle
