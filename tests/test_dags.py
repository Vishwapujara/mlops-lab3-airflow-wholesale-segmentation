"""DAG integrity checks plus full runs of both DAGs with dag.test()."""

import json

import pytest
from airflow.dag_processing.dagbag import DagBag

from wholesale_lib.assets import CLEAN_DATA
from wholesale_lib.config import ALERTS_DIR, CLEAN_PATH, QUARANTINE_DIR, RUNS_DIR


@pytest.fixture(scope="module")
def dagbag():
    from tests.conftest import DAGS
    return DagBag(dag_folder=str(DAGS))


def test_no_import_errors(dagbag):
    assert dagbag.import_errors == {}
    assert set(dagbag.dag_ids) == {"wholesale_ingest", "wholesale_segmentation"}


def test_ingest_structure(dagbag):
    dag = dagbag.get_dag("wholesale_ingest")
    assert dag.get_task("route").downstream_task_ids == {"publish_clean", "quarantine"}
    assert dag.get_task("fetch_raw").retries == 3
    assert [o.name for o in dag.get_task("publish_clean").outlets] == [CLEAN_DATA.name]


def test_segmentation_is_scheduled_on_the_clean_data_asset(dagbag):
    dag = dagbag.get_dag("wholesale_segmentation")
    assert CLEAN_DATA.name in str(dag.timetable.asset_condition)
    assert dag.get_task("model_selection.train_kmeans").is_mapped


@pytest.mark.parametrize("dag_id", ["wholesale_ingest", "wholesale_segmentation"])
def test_every_task_has_retries_and_failure_alert(dagbag, dag_id):
    for task in dagbag.get_dag(dag_id).tasks:
        assert task.retries >= 1, task.task_id
        assert task.on_failure_callback, task.task_id


def states(run):
    return {ti.task_id: ti.state for ti in run.get_task_instances()}


def test_bad_data_is_quarantined_and_alerts():
    from wholesale_ingest import dag
    run = dag.test(run_conf={"inject_bad_rows": True, "use_bundled_copy": True})
    assert run.state == "failed"
    assert states(run) == {"fetch_raw": "success", "validate": "success", "route": "success",
                           "publish_clean": "skipped", "quarantine": "failed"}
    assert not CLEAN_PATH.exists()
    assert list(QUARANTINE_DIR.glob("bad_*.csv"))
    alert = json.loads(next(ALERTS_DIR.glob("wholesale_ingest__quarantine__*.json")).read_text())
    assert "Data quality failed" in alert["error"]


def test_good_data_is_published_then_segmented():
    from wholesale_ingest import dag as ingest
    run = ingest.test(run_conf={"inject_bad_rows": False, "use_bundled_copy": True})
    assert run.state == "success" and states(run)["quarantine"] == "skipped"
    assert CLEAN_PATH.exists()

    from wholesale_segmentation import dag as segmentation
    run = segmentation.test(run_conf={"k_min": 2, "k_max": 5})
    assert run.state == "success"
    mapped = [ti for ti in run.get_task_instances() if ti.task_id == "model_selection.train_kmeans"]
    assert len(mapped) == 4  # one task per k in 2..5
    summary = json.loads(next(RUNS_DIR.glob("*/summary.json")).read_text())
    assert 2 <= summary["chosen_k"] <= 5
    out = RUNS_DIR / next(RUNS_DIR.iterdir()).name
    assert {"segment_profiles.csv", "customer_segments.csv", "model_selection.png"} <= {p.name for p in out.iterdir()}
