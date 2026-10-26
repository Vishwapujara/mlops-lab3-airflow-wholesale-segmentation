"""Point Airflow and the DAGs at throwaway folders before anything imports them."""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DAGS = ROOT / "dags"
TMP = Path(tempfile.mkdtemp(prefix="wholesale-tests-"))

os.environ["AIRFLOW_HOME"] = str(TMP / "airflow")
os.environ["AIRFLOW__CORE__DAGS_FOLDER"] = str(DAGS)
os.environ["AIRFLOW__CORE__LOAD_EXAMPLES"] = "False"
os.environ["WHOLESALE_WORKING_DIR"] = str(TMP / "working_data")
sys.path.insert(0, str(DAGS))

# dag.test() needs a metadata database; a fresh SQLite one takes a few seconds to create.
subprocess.run([sys.executable, "-m", "airflow", "db", "migrate"], check=True, capture_output=True)
