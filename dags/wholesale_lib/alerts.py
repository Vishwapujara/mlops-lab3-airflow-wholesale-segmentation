"""Failure callback: writes an alert file instead of sending email (no SMTP account needed)."""

import json
from datetime import datetime, timezone

from wholesale_lib.config import ALERTS_DIR


def on_task_failure(context):
    ti = context["ti"]
    alert = {
        "time": datetime.now(timezone.utc).isoformat(),
        "dag_id": ti.dag_id,
        "task_id": ti.task_id,
        "run_id": context["run_id"],
        "try_number": ti.try_number,
        "error": repr(context.get("exception")),
    }
    ALERTS_DIR.mkdir(parents=True, exist_ok=True)
    path = ALERTS_DIR / f"{ti.dag_id}__{ti.task_id}__{alert['time'][:19].replace(':', '-')}.json"
    path.write_text(json.dumps(alert, indent=2))
    print(f"ALERT written to {path}: {alert['error']}")
