"""Paths and column names shared by both DAGs."""

import os
from pathlib import Path

DATA_URL = "https://archive.ics.uci.edu/static/public/292/wholesale+customers.zip"
CSV_IN_ZIP = "Wholesale customers data.csv"
BUNDLED_CSV = Path(__file__).resolve().parent.parent / "data" / "wholesale_customers.csv"

# Inside Docker this folder is mounted from ./working_data on the host.
WORKING_DIR = Path(os.environ.get("WHOLESALE_WORKING_DIR", "/opt/airflow/working_data"))
RAW_DIR = WORKING_DIR / "raw"
CLEAN_PATH = WORKING_DIR / "clean" / "wholesale_customers.csv"
QUARANTINE_DIR = WORKING_DIR / "quarantine"
RUNS_DIR = WORKING_DIR / "segmentation"
ALERTS_DIR = WORKING_DIR / "alerts"

# Annual spend (monetary units) per product category.
SPEND_COLUMNS = ["Fresh", "Milk", "Grocery", "Frozen", "Detergents_Paper", "Delicassen"]
CHANNELS = {1: "Horeca", 2: "Retail"}      # Horeca = hotel / restaurant / cafe
REGIONS = {1: "Lisbon", 2: "Oporto", 3: "Other"}
EXPECTED_COLUMNS = ["Channel", "Region"] + SPEND_COLUMNS


def run_dir(run_id):
    """Per-run output folder, so runs never overwrite each other."""
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in run_id)
    path = RUNS_DIR / safe
    path.mkdir(parents=True, exist_ok=True)
    return path
