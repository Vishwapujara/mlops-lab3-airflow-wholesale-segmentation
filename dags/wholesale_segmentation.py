"""
### wholesale_segmentation

Segment wholesale customers with K-Means. Runs automatically whenever `wholesale_ingest`
publishes new validated data (it is scheduled on the `wholesale_clean_data` asset), or manually
with your own params.

`prepare_features` → **model_selection** task group (`candidate_ks` → one mapped `train_kmeans`
task per k, run in parallel → `select_k`) → `profile_segments`.

Outputs go to `working_data/segmentation/<run_id>/`: `summary.json`, `segment_profiles.csv`,
`customer_segments.csv`, `model_selection.png` and one model per k in `models/`.
"""

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from airflow.sdk import Param, dag, get_current_context, task, task_group

from wholesale_lib import clustering
from wholesale_lib.alerts import on_task_failure
from wholesale_lib.assets import CLEAN_DATA
from wholesale_lib.config import CLEAN_PATH, SPEND_COLUMNS, run_dir


@dag(
    dag_id="wholesale_segmentation",
    schedule=[CLEAN_DATA],
    start_date=datetime(2026, 10, 1),
    catchup=False,
    max_active_runs=1,
    default_args={
        "owner": "Vishwa Pujara",
        "retries": 1,
        "retry_delay": timedelta(seconds=30),
        "execution_timeout": timedelta(minutes=10),
        "on_failure_callback": on_task_failure,
    },
    params={
        "k_min": Param(2, type="integer", minimum=2, description="Smallest number of segments to try"),
        "k_max": Param(10, type="integer", minimum=3, maximum=20, description="Largest number of segments to try"),
        "features": Param(SPEND_COLUMNS, type="array", items={"type": "string", "enum": SPEND_COLUMNS},
                          description="Spend categories to cluster on"),
        "selection_metric": Param("silhouette", type="string", enum=["silhouette", "elbow"],
                                  description="How to choose k"),
    },
    tags=["lab3", "ml"],
    doc_md=__doc__,
)
def wholesale_segmentation():

    @task
    def prepare_features(params=None, run_id=None) -> dict:
        df = pd.read_csv(CLEAN_PATH)
        X, rows, report = clustering.prepare_features(df, params["features"])
        out = run_dir(run_id)
        np.save(out / "features.npy", X)
        rows.to_csv(out / "rows.csv", index=False)
        print(f"{report['rows_kept']} of {report['rows_in']} customers kept "
              f"({report['outliers_removed']} extreme outliers removed), features: {report['features']}")
        return {**report, "dir": str(out)}

    @task_group(group_id="model_selection")
    def model_selection(prep: dict):

        @task
        def candidate_ks(params=None) -> list[int]:
            if params["k_min"] >= params["k_max"]:
                raise ValueError(f"k_min ({params['k_min']}) must be smaller than k_max ({params['k_max']})")
            return list(range(params["k_min"], params["k_max"] + 1))

        # One task instance per k, run in parallel; the UI labels each one "k=<value>".
        @task(map_index_template="k={{ k_label }}")
        def train_kmeans(k: int, prep: dict) -> dict:
            get_current_context()["k_label"] = k
            model, metrics = clustering.train_kmeans(np.load(Path(prep["dir"]) / "features.npy"), k)
            models_dir = Path(prep["dir"]) / "models"
            models_dir.mkdir(exist_ok=True)
            model_path = models_dir / f"kmeans_k{k}.pkl"
            clustering.save_pickle(model, model_path)
            print(f"k={k}: SSE={metrics['sse']:.1f}, silhouette={metrics['silhouette']:.4f}")
            return {**metrics, "model_path": str(model_path)}

        @task
        def select_k(results: list[dict], params=None) -> dict:
            results = list(results)
            selection = clustering.select_k(results, params["selection_metric"])
            chosen = next(r for r in results if r["k"] == selection["chosen_k"])
            print(f"Chosen k={selection['chosen_k']} by {selection['metric']} "
                  f"(silhouette suggests {selection['silhouette_k']}, elbow suggests {selection['elbow_k']})")
            return {**selection, "model_path": chosen["model_path"], "results": results}

        return select_k(train_kmeans.partial(prep=prep).expand(k=candidate_ks()))

    @task
    def profile_segments(prep: dict, selection: dict) -> dict:
        out = Path(prep["dir"])
        rows = pd.read_csv(out / "rows.csv")
        labels = clustering.load_pickle(selection["model_path"]).labels_
        profile = clustering.profile_segments(rows, labels, prep["features"])
        profile.to_csv(out / "segment_profiles.csv", index=False)
        rows.assign(segment=labels).to_csv(out / "customer_segments.csv", index=False)
        clustering.selection_figure(selection["results"], selection).savefig(out / "model_selection.png", dpi=120)

        chosen = next(r for r in selection["results"] if r["k"] == selection["chosen_k"])
        summary = {
            "chosen_k": selection["chosen_k"], "selection_metric": selection["metric"],
            "silhouette": round(chosen["silhouette"], 4),
            "silhouette_k": selection["silhouette_k"], "elbow_k": selection["elbow_k"],
            "channel_agreement_ari": round(clustering.channel_agreement(rows, labels), 4),
            "customers": prep["rows_kept"], "outliers_removed": prep["outliers_removed"],
            "features": prep["features"], "output_dir": str(out),
        }
        clustering.write_json(summary, out / "summary.json")
        print(profile.to_string(index=False))
        print(summary)
        return summary

    prep = prepare_features()
    profile_segments(prep, model_selection(prep))


dag = wholesale_segmentation()

if __name__ == "__main__":
    dag.test()
