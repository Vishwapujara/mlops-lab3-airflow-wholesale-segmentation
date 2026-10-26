import numpy as np
import pandas as pd

from wholesale_lib import quality


def failed(df):
    return quality.summary(quality.run_checks(df))["failed_checks"]


def test_bundled_data_passes_every_check():
    assert failed(quality.load_bundled()) == []


def test_injected_bad_rows_are_caught():
    assert failed(quality.inject_bad_rows(quality.load_bundled())) == ["no_nulls", "non_negative_spend"]


def test_missing_column_stops_at_schema_check():
    checks = quality.run_checks(quality.load_bundled().drop(columns=["Milk"]))
    assert [c["check"] for c in checks] == ["schema"]
    assert not checks[0]["passed"]


def test_invalid_codes_and_too_few_rows():
    df = quality.load_bundled().head(50).copy()
    df.loc[0, "Channel"] = 3
    df.loc[1, "Region"] = 0
    assert failed(df) == ["min_rows", "valid_channel", "valid_region"]


def test_duplicates_over_limit():
    df = quality.load_bundled()
    df = pd.concat([df, df.sample(40, random_state=0)], ignore_index=True)
    assert failed(df) == ["few_duplicates"]


def test_inject_is_deterministic():
    a = quality.inject_bad_rows(quality.load_bundled())
    b = quality.inject_bad_rows(quality.load_bundled())
    assert a.equals(b)
    assert np.isnan(a["Grocery"]).sum() == 2 and (a["Milk"] < 0).sum() == 3
