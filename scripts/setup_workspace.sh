#!/usr/bin/env bash
set -euo pipefail

mkdir -p data/raw
mkdir -p data/processed
mkdir -p data/external
mkdir -p checkpoints
mkdir -p logs
mkdir -p outputs/reports
mkdir -p outputs/exports
mkdir -p outputs/figures
mkdir -p notebooks
mkdir -p docs/architecture
mkdir -p docs/runbooks
mkdir -p artifacts/models

printf "ZANE workspace folders are ready.\n"
