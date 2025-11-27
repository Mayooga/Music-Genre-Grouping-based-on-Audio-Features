"""
End-to-end unsupervised clustering pipeline for music audio features.

Usage example:
    python src/genre_cluster.py --data data/audio_features.csv --cluster-algo kmeans --n-clusters 12
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


@dataclass
class ClusterConfig:
    data_path: Path
    output_dir: Path
    cluster_algo: str = "kmeans"
    n_clusters: int = 8
    pca_components: Optional[int] = None
    label_col: Optional[str] = None
    random_state: int = 42
    dbscan_eps: float = 0.7
    dbscan_min_samples: int = 15


def parse_args() -> ClusterConfig:
    parser = argparse.ArgumentParser(
        description="Cluster tracks into genre-like groups using audio features."
    )
    parser.add_argument("--data", required=True, help="Path to CSV with audio features.")
    parser.add_argument(
        "--output-dir",
        default="artifacts",
        help="Folder for clustered data + reports (created if missing).",
    )
    parser.add_argument(
        "--cluster-algo",
        default="kmeans",
        choices=["kmeans", "gmm", "agglomerative", "dbscan"],
        help="Unsupervised algorithm to apply.",
    )
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=8,
        help="Target number of clusters (ignored by DBSCAN).",
    )
    parser.add_argument(
        "--pca-components",
        type=int,
        default=None,
        help="Optional PCA dimension before clustering.",
    )
    parser.add_argument(
        "--label-col",
        default=None,
        help="Optional column with existing genre labels for cross-tab analysis.",
    )
    parser.add_argument(
        "--random-state", type=int, default=42, help="Reproducibility seed."
    )
    parser.add_argument(
        "--dbscan-eps",
        type=float,
        default=0.7,
        help="Neighborhood radius for DBSCAN (only used when cluster-algo=dbscan).",
    )
    parser.add_argument(
        "--dbscan-min-samples",
        type=int,
        default=15,
        help="Min samples per core point for DBSCAN.",
    )

    args = parser.parse_args()
    return ClusterConfig(
        data_path=Path(args.data),
        output_dir=Path(args.output_dir),
        cluster_algo=args.cluster_algo.lower(),
        n_clusters=args.n_clusters,
        pca_components=args.pca_components,
        label_col=args.label_col,
        random_state=args.random_state,
        dbscan_eps=args.dbscan_eps,
        dbscan_min_samples=args.dbscan_min_samples,
    )


def require_numeric_features(df: pd.DataFrame, label_col: Optional[str]) -> Tuple[pd.DataFrame, List[str]]:
    """Extract numeric feature columns, ensuring at least two exist."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if label_col and label_col in numeric_cols:
        numeric_cols.remove(label_col)

    if len(numeric_cols) < 2:
        raise ValueError(
            "Need at least two numeric feature columns for clustering. "
            "Check your dataset or feature engineering step."
        )
    return df[numeric_cols].copy(), numeric_cols


def preprocess_features(features: pd.DataFrame) -> np.ndarray:
    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    return pipeline.fit_transform(features)


def reduce_dimensions(
    matrix: np.ndarray, n_components: Optional[int], random_state: int
) -> np.ndarray:
    if not n_components or n_components >= matrix.shape[1]:
        return matrix
    reducer = PCA(n_components=n_components, random_state=random_state)
    return reducer.fit_transform(matrix)


def build_model(cfg: ClusterConfig):
    algo = cfg.cluster_algo
    if algo == "kmeans":
        return KMeans(
            n_clusters=cfg.n_clusters, n_init="auto", random_state=cfg.random_state
        )
    if algo == "gmm":
        return GaussianMixture(
            n_components=cfg.n_clusters, covariance_type="full", random_state=cfg.random_state
        )
    if algo == "agglomerative":
        return AgglomerativeClustering(n_clusters=cfg.n_clusters)
    if algo == "dbscan":
        return DBSCAN(eps=cfg.dbscan_eps, min_samples=cfg.dbscan_min_samples)
    raise ValueError(f"Unsupported cluster algorithm: {algo}")


def fit_predict(model, matrix: np.ndarray) -> np.ndarray:
    if isinstance(model, GaussianMixture):
        return model.fit_predict(matrix)
    return model.fit_predict(matrix)


def compute_metrics(
    cfg: ClusterConfig, model, matrix: np.ndarray, labels: np.ndarray
) -> Dict[str, float]:
    unique_labels = set(labels.tolist())
    n_clusters = len(unique_labels - {-1})
    metrics: Dict[str, float] = {
        "n_samples": len(labels),
        "n_clusters": n_clusters,
    }

    if len(unique_labels) > 1 and len(unique_labels - {-1}) > 0:
        try:
            metrics["silhouette_score"] = float(silhouette_score(matrix, labels))
        except ValueError:
            pass

    if cfg.cluster_algo == "kmeans" and hasattr(model, "inertia_"):
        metrics["inertia"] = float(model.inertia_)
    if cfg.cluster_algo == "gmm" and hasattr(model, "lower_bound_"):
        metrics["avg_log_likelihood"] = float(model.lower_bound_)

    counts = pd.Series(labels).value_counts().to_dict()
    metrics["cluster_sizes"] = {str(k): int(v) for k, v in counts.items()}
    return metrics


def save_outputs(
    df: pd.DataFrame,
    features: List[str],
    labels: np.ndarray,
    metrics: Dict[str, float],
    cfg: ClusterConfig,
) -> None:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    labeled_df = df.copy()
    labeled_df["cluster_label"] = labels

    clustered_path = cfg.output_dir / "clustered_audio_features.csv"
    labeled_df.to_csv(clustered_path, index=False)

    summary = (
        labeled_df.groupby("cluster_label")[features]
        .mean()
        .round(4)
    )
    summary["count"] = labeled_df.groupby("cluster_label").size()
    summary.to_csv(cfg.output_dir / "cluster_summary.csv")

    if cfg.label_col and cfg.label_col in df.columns:
        crosstab = pd.crosstab(df[cfg.label_col], labels, normalize="index")
        crosstab.to_csv(cfg.output_dir / "label_cluster_crosstab.csv")

    metrics_path = cfg.output_dir / "cluster_report.json"
    with metrics_path.open("w", encoding="utf-8") as fp:
        json.dump(metrics, fp, indent=2)


def main() -> None:
    cfg = parse_args()
    if not cfg.data_path.exists():
        raise FileNotFoundError(f"Dataset not found at {cfg.data_path}")

    df = pd.read_csv(cfg.data_path)
    feature_df, feature_cols = require_numeric_features(df, cfg.label_col)

    processed = preprocess_features(feature_df)
    reduced = reduce_dimensions(processed, cfg.pca_components, cfg.random_state)

    model = build_model(cfg)
    labels = fit_predict(model, reduced)
    metrics = compute_metrics(cfg, model, reduced, labels)

    save_outputs(df, feature_cols, labels, metrics, cfg)
    print(f"Done. Saved clustered dataset + reports to: {cfg.output_dir.resolve()}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

