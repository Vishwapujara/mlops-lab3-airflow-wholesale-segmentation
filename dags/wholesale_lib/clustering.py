"""Feature preparation, K-Means training, model selection and segment profiling."""

import json
import pickle

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from kneed import KneeLocator
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from wholesale_lib.config import CHANNELS, SPEND_COLUMNS

SEED = 42


def prepare_features(df, features, outlier_iqr=1.5):
    """Log-transform spend (it is heavily right-skewed), drop extreme outliers, standardise.

    Returns the scaled matrix, the kept rows of the original frame, and a small report.
    """
    unknown = [f for f in features if f not in SPEND_COLUMNS]
    if unknown:
        raise ValueError(f"Unknown features {unknown}; choose from {SPEND_COLUMNS}")
    logged = np.log1p(df[features])
    q1, q3 = logged.quantile(0.25), logged.quantile(0.75)
    iqr = q3 - q1
    # A customer is dropped only if it is an outlier in 2+ categories (Tukey fences on the log scale).
    outlier_counts = ((logged < q1 - outlier_iqr * iqr) | (logged > q3 + outlier_iqr * iqr)).sum(axis=1)
    keep = outlier_counts < 2
    scaled = StandardScaler().fit_transform(logged[keep])
    report = {"rows_in": int(len(df)), "rows_kept": int(keep.sum()),
              "outliers_removed": int((~keep).sum()), "features": list(features)}
    return scaled, df[keep].reset_index(drop=True), report


def train_kmeans(X, k, seed=SEED):
    model = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(X)
    return model, {"k": k, "sse": float(model.inertia_),
                   "silhouette": float(silhouette_score(X, model.labels_))}


def select_k(results, metric="silhouette"):
    """Pick k by best silhouette, or by the elbow of the SSE curve (as in the original lab)."""
    results = sorted(results, key=lambda r: r["k"])
    ks = [r["k"] for r in results]
    elbow = KneeLocator(ks, [r["sse"] for r in results], curve="convex", direction="decreasing").elbow
    best_silhouette = max(results, key=lambda r: r["silhouette"])["k"]
    if metric == "silhouette":
        chosen = best_silhouette
    elif metric == "elbow":
        chosen = elbow if elbow is not None else best_silhouette
    else:
        raise ValueError(f"metric must be 'silhouette' or 'elbow', got {metric!r}")
    return {"chosen_k": int(chosen), "metric": metric,
            "silhouette_k": int(best_silhouette), "elbow_k": None if elbow is None else int(elbow)}


def profile_segments(rows, labels, features):
    """Average annual spend per segment in original units, plus size and channel mix."""
    frame = rows.assign(segment=labels)
    profile = frame.groupby("segment")[features].mean().round(0)
    profile.insert(0, "customers", frame.groupby("segment").size())
    profile["retail_share"] = frame.groupby("segment")["Channel"].apply(lambda s: (s == 2).mean()).round(2)
    profile["top_category"] = profile[features].idxmax(axis=1)
    return profile.reset_index()


def channel_agreement(rows, labels):
    """Adjusted Rand index between segments and the known Horeca / Retail channel (0 = unrelated, 1 = identical)."""
    return float(adjusted_rand_score(rows["Channel"].map(CHANNELS), labels))


def selection_figure(results, selection):
    results = sorted(results, key=lambda r: r["k"])
    ks = [r["k"] for r in results]
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4))
    left.plot(ks, [r["sse"] for r in results], marker="o")
    if selection["elbow_k"]:
        left.axvline(selection["elbow_k"], ls="--", color="grey", label=f"elbow k={selection['elbow_k']}")
        left.legend()
    left.set(title="Elbow method (SSE)", xlabel="k", ylabel="Within-cluster SSE")
    right.plot(ks, [r["silhouette"] for r in results], marker="o", color="tab:orange")
    right.axvline(selection["chosen_k"], ls="--", color="grey", label=f"chosen k={selection['chosen_k']}")
    right.set(title="Silhouette score", xlabel="k", ylabel="Silhouette")
    right.legend()
    fig.tight_layout()
    return fig


def save_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def write_json(obj, path):
    path.write_text(json.dumps(obj, indent=2, default=str))
