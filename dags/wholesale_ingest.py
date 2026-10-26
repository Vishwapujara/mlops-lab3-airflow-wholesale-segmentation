"""
### wholesale_ingest

Fetch the UCI Wholesale Customers data, run data quality checks, and publish it.

`fetch_raw` → `validate` → `route` (branch) → **`publish_clean`** (updates the `wholesale_clean_data` asset,
which triggers `wholesale_segmentation`) **or** `quarantine` (keeps the bad file and fails the run).

Params: `inject_bad_rows` corrupts a few rows to show the quarantine path;
`use_bundled_copy` skips the download and uses `dags/data/wholesale_customers.csv`.
"""

import shutil
from datetime import datetime, timedelta

import pandas as pd
import requests
from airflow.sdk import Param, dag, get_current_context, task
from airflow.sdk.exceptions import AirflowFailException

from wholesale_lib import quality
from wholesale_lib.alerts import on_task_failure
from wholesale_lib.assets import CLEAN_DATA
from wholesale_lib.clustering import write_json
from wholesale_lib.config import CLEAN_PATH, QUARANTINE_DIR, RAW_DIR


@dag(
    dag_id="wholesale_ingest",
    schedule="@weekly",
    start_date=datetime(2026, 10, 1),
    catchup=False,
    max_active_runs=1,
    default_args={
        "owner": "Vishwa Pujara",
        "retries": 1,
        "retry_delay": timedelta(seconds=30),
        "execution_timeout": timedelta(minutes=5),
        "on_failure_callback": on_task_failure,
    },
    params={
        "inject_bad_rows": Param(False, type="boolean",
                                 description="Corrupt a few rows to demonstrate the quarantine branch"),
        "use_bundled_copy": Param(False, type="boolean",
                                  description="Skip the download and use the copy in dags/data/"),
    },
    tags=["lab3", "ingest"],
    doc_md=__doc__,
)
def wholesale_ingest():

    @task(retries=3, retry_delay=timedelta(seconds=10), retry_exponential_backoff=True)
    def fetch_raw(params=None, run_id=None) -> dict:
        """Download from UCI. Network errors are retried; on the last try fall back to the bundled copy."""
        if params["use_bundled_copy"]:
            df, source = quality.load_bundled(), "bundled copy"
        else:
            try:
                df, source = quality.download(), "UCI download"
            except requests.RequestException as exc:
                context = get_current_context()
                if context["ti"].try_number <= context["task"].retries:
                    raise  # let Airflow retry with backoff
                print(f"Download failed on the last try ({exc!r}); using the bundled copy")
                df, source = quality.load_bundled(), "bundled copy (download failed)"
        if params["inject_bad_rows"]:
            df = quality.inject_bad_rows(df)
            source += " + injected bad rows"
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        path = RAW_DIR / f"{run_id.replace(':', '-').replace('+', '_')}.csv"
        df.to_csv(path, index=False)
        print(f"Saved {len(df)} rows from {source} to {path}")
        return {"path": str(path), "source": source, "rows": len(df)}

    @task
    def validate(raw: dict) -> dict:
        report = quality.summary(quality.run_checks(pd.read_csv(raw["path"])))
        for check in report["checks"]:
            print(f"[{'PASS' if check['passed'] else 'FAIL'}] {check['check']}: {check['detail']}")
        return {**raw, **report}

    @task.branch
    def route(report: dict) -> str:
        return "publish_clean" if report["passed"] else "quarantine"

    @task(outlets=[CLEAN_DATA])
    def publish_clean(report: dict) -> str:
        CLEAN_PATH.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(report["path"], CLEAN_PATH)
        # Extra info attached to this asset event, visible in the Assets view.
        get_current_context()["outlet_events"][CLEAN_DATA].extra = {
            "rows": report["rows"], "source": report["source"]}
        print(f"Published {report['rows']} validated rows to {CLEAN_PATH}")
        return str(CLEAN_PATH)

    @task
    def quarantine(report: dict):
        QUARANTINE_DIR.mkdir(parents=True, exist_ok=True)
        kept = QUARANTINE_DIR / ("bad_" + report["path"].rsplit("/", 1)[-1])
        shutil.copyfile(report["path"], kept)
        write_json(report, kept.with_suffix(".report.json"))
        raise AirflowFailException(
            f"Data quality failed ({', '.join(report['failed_checks'])}). File kept at {kept}; "
            "the clean dataset was NOT updated, so segmentation does not run.")

    report = validate(fetch_raw())
    route(report) >> [publish_clean(report), quarantine(report)]


dag = wholesale_ingest()

if __name__ == "__main__":
    dag.test()
