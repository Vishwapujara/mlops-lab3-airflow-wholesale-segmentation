import numpy as np
import pytest

from wholesale_lib import clustering
from wholesale_lib.config import SPEND_COLUMNS
from wholesale_lib.quality import load_bundled


@pytest.fixture(scope="module")
def prepared():
    return clustering.prepare_features(load_bundled(), SPEND_COLUMNS)


def test_prepare_features_scales_and_drops_outliers(prepared):
    X, rows, report = prepared
    assert X.shape == (report["rows_kept"], len(SPEND_COLUMNS)) == (len(rows), 6)
    assert report["rows_in"] == 440 and 0 < report["outliers_removed"] < 20
    assert np.allclose(X.mean(axis=0), 0, atol=1e-9) and np.allclose(X.std(axis=0), 1)


def test_feature_subset_and_unknown_feature():
    X, _, _ = clustering.prepare_features(load_bundled(), ["Milk", "Grocery"])
    assert X.shape[1] == 2
    with pytest.raises(ValueError, match="Unknown features"):
        clustering.prepare_features(load_bundled(), ["Milk", "Wine"])


def test_train_and_select(prepared):
    X, _, _ = prepared
    results = [clustering.train_kmeans(X, k)[1] for k in range(2, 8)]
    sse = [r["sse"] for r in results]
    assert sse == sorted(sse, reverse=True)  # more clusters, lower SSE
    by_silhouette = clustering.select_k(results, "silhouette")
    assert by_silhouette["chosen_k"] == max(results, key=lambda r: r["silhouette"])["k"]
    assert clustering.select_k(results, "elbow")["chosen_k"] == by_silhouette["elbow_k"]
    with pytest.raises(ValueError):
        clustering.select_k(results, "gap")


def test_profile_and_channel_agreement(prepared):
    X, rows, _ = prepared
    model, _ = clustering.train_kmeans(X, 2)
    profile = clustering.profile_segments(rows, model.labels_, SPEND_COLUMNS)
    assert profile["customers"].sum() == len(rows)
    assert set(profile["top_category"]) <= set(SPEND_COLUMNS)
    # Two segments on this data largely recover the known Horeca / Retail channel.
    assert clustering.channel_agreement(rows, model.labels_) > 0.3
