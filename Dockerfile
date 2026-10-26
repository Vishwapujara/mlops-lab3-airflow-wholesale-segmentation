# Airflow image with this lab's Python packages baked in, instead of reinstalling them on
# every container start with _PIP_ADDITIONAL_REQUIREMENTS (which the official compose file warns against).
FROM apache/airflow:3.3.2

COPY requirements.txt /requirements.txt
# Install against Airflow's official constraints so nothing upgrades or breaks Airflow itself.
RUN PY=$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")') && \
    pip install --no-cache-dir "apache-airflow==${AIRFLOW_VERSION}" -r /requirements.txt \
      --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-${PY}.txt"
