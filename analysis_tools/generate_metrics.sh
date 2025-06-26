#!/usr/bin/env bash
set -euo pipefail

WORKDIR=$1
KS_DIR="$WORKDIR/KS_EMD"
SCRIPTS_DIR="$KS_DIR/scripts"
RESULTS_DIR="$KS_DIR/results"
OUT_DIR="$WORKDIR/metrics"

if [[ ! -d "$KS_DIR" ]]; then
  echo "KS directory not found: $KS_DIR"
  exit 1
fi
mkdir -p "$SCRIPTS_DIR"
mkdir -p "$OUT_DIR"
mkdir -p "$RESULTS_DIR"

# determine prefix from first piece
first_pkl=$(ls "$KS_DIR"/*.piece0.pkl | head -n1)
FILEPREFIX=$(basename "$first_pkl" | cut -d '.' -f1)

# count pieces
nparts=$(ls "$KS_DIR"/*.pkl | wc -l)

# Loop through parts and create scripts, then qsub them
for i in $(seq 0 $((nparts-1))); do
    script_path="$SCRIPTS_DIR/run_test_${i}.sh"
    (
        echo "#!/bin/bash"
        echo "source activate pycytominer"
        echo "python3 perform_KS_test.py $KS_DIR/${FILEPREFIX}.groupedvariants.piece${i}.pkl $KS_DIR/${FILEPREFIX}.WT.piece${i}.csv"
    ) > "$script_path"

    chmod u+x "$script_path"
    qsub -l mfree=20G "$script_path"
done
