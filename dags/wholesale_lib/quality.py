"""Data quality checks run before any data is published for training."""

import io
import zipfile

import numpy as np
import pandas as pd
import requests

from wholesale_lib.config import (BUNDLED_CSV, CHANNELS, CSV_IN_ZIP, DATA_URL, EXPECTED_COLUMNS,
                                  REGIONS, SPEND_COLUMNS)

MIN_ROWS = 100
MAX_DUPLICATE_SHARE = 0.05


def download(url=DATA_URL, timeout=30):
    """Fetch the dataset from UCI. Raises on network errors so Airflow can retry."""
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
        return pd.read_csv(zf.open(CSV_IN_ZIP))


def load_bundled():
    return pd.read_csv(BUNDLED_CSV)


def inject_bad_rows(df, n=5, seed=0):
    """Corrupt a few rows on purpose, to demonstrate the quarantine branch."""
    df = df.copy()
    rows = np.random.default_rng(seed).choice(len(df), size=n, replace=False)
    df.loc[df.index[rows[: n // 2 + 1]], "Milk"] = -500
    df.loc[df.index[rows[n // 2 + 1:]], "Grocery"] = np.nan
    return df


def run_checks(df):
    """Return a list of {check, passed, detail} dicts. Every check runs, even after a failure."""
    checks = []

    def add(name, passed, detail):
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    missing_cols = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    add("schema", not missing_cols, f"missing columns: {missing_cols}" if missing_cols else "all 8 columns present")
    if missing_cols:
        return checks

    add("min_rows", len(df) >= MIN_ROWS, f"{len(df)} rows (need >= {MIN_ROWS})")
    nulls = int(df[EXPECTED_COLUMNS].isna().sum().sum())
    add("no_nulls", nulls == 0, f"{nulls} missing values")
    negatives = int((df[SPEND_COLUMNS] < 0).sum().sum())
    add("non_negative_spend", negatives == 0, f"{negatives} negative spend values")
    bad_channel = int((~df["Channel"].isin(CHANNELS)).sum())
    add("valid_channel", bad_channel == 0, f"{bad_channel} rows outside {sorted(CHANNELS)}")
    bad_region = int((~df["Region"].isin(REGIONS)).sum())
    add("valid_region", bad_region == 0, f"{bad_region} rows outside {sorted(REGIONS)}")
    dup_share = float(df.duplicated().mean())
    add("few_duplicates", dup_share <= MAX_DUPLICATE_SHARE, f"{dup_share:.1%} duplicate rows")
    return checks


def summary(checks):
    failed = [c["check"] for c in checks if not c["passed"]]
    return {"passed": not failed, "failed_checks": failed, "checks": checks}
