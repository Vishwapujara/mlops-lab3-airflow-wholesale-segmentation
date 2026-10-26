"""The Airflow Asset that connects the two DAGs."""

from airflow.sdk import Asset

from wholesale_lib.config import CLEAN_PATH

# wholesale_ingest updates this asset when it publishes validated data;
# wholesale_segmentation is scheduled on it, so it runs automatically after every successful ingest.
CLEAN_DATA = Asset(name="wholesale_clean_data", uri=f"file://{CLEAN_PATH}")
