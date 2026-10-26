#!/bin/bash
# One-time setup: create the folders Airflow mounts and write .env (your user ID + an encryption key).
set -e
mkdir -p ./logs ./plugins ./config ./working_data
# Keep an existing key on re-runs: Airflow encrypts stored connections/variables with it.
FERNET_KEY=$(grep -s '^FERNET_KEY=' .env | cut -d= -f2-)
[ -z "$FERNET_KEY" ] && FERNET_KEY=$(openssl rand -base64 32 | tr '+/' '-_')
printf 'AIRFLOW_UID=%s\nFERNET_KEY=%s\n' "$(id -u)" "$FERNET_KEY" > .env
echo "Created logs/ plugins/ config/ working_data/ and .env (AIRFLOW_UID=$(id -u), FERNET_KEY set)"
