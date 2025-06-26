#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<EOF
Usage: $0 EXPERIMENT BARCODE_MAP CELL_FEATURES \
       [--metadata-columns-file META_FILE] \
       [--blacklist-grep-file GREP_FILE] \
       [--bc-threshold N] \
       [--variant-bc-threshold N] \
       [--normalize-to (all|synonymous)] \
       [--barcode-name (virtualBarcode|upBarcode)]

  EXPERIMENT              Name of the experiment (e.g. PTENT3R1.neuron)
  CELL_FEATURES           Path to cell‐level features (CSV or Parquet)

  --metadata-columns-file File with one metadata column per line (optional)
  --blacklist-grep-file   File with one grep pattern per line (optional)
  --bc-threshold          Min cells per barcode (default: 10)
  --variant-bc-threshold  Min barcodes per variant (default: 4)
  --normalize-to          Normalize to 'all' variants or only 'synonymous' (default: all)
  --barcode-name          What is the name of the barcode? (default: virtualBarcode)
EOF
  exit 1
}

# require at least the 3 positional args
if [ "$#" -lt 2 ]; then
  usage
fi

EXPERIMENT=$1
CELL_FEATURES=$2
shift 2

# default optional values
META_FILE=""
BLACKLIST_GREP_FILE=""
BC_THRESHOLD=10
VARIANT_BC_THRESHOLD=4
NORMALIZE_TO="all"
BARCODE_NAME="virtualBarcode"

# parse flags
while (( "$#" )); do
  case "$1" in
    --metadata-columns-file)
      META_FILE=$2
      shift 2
      ;;

    --blacklist-grep-file)
      BLACKLIST_GREP_FILE=$2
      shift 2
      ;;

    --bc-threshold)
      BC_THRESHOLD=$2
      shift 2
      ;;

    --variant-bc-threshold)
      VARIANT_BC_THRESHOLD=$2
      shift 2
      ;;

    --normalize-to)
      NORMALIZE_TO=$2
      shift 2
      ;;

    --barcode-name)
      BARCODE_NAME=$2
      shift 2
      ;;

    *)
      echo "Unknown option: $1"
      usage
      ;;
  esac
done

# prepare working directory
WORKDIR=outputs/${EXPERIMENT}
mkdir -p "${WORKDIR}"

# 1) build aggregation command
cmd=(python3 analysis_tools/aggregate_features.py \
  --experiment           "${EXPERIMENT}" \
  --cell-features        "${CELL_FEATURES}" \
  --output-dir           "${WORKDIR}" \
  --bc-threshold         "${BC_THRESHOLD}" \
  --variant-bc-threshold "${VARIANT_BC_THRESHOLD}"
  --barcode-name         "${BARCODE_NAME}")

# include optional metadata-columns-file
if [ -n "${META_FILE}" ]; then
  cmd+=(--metadata-columns-file "${META_FILE}")
fi

# include optional blacklist-grep-file
if [ -n "${BLACKLIST_GREP_FILE}" ]; then
  cmd+=(--blacklist-grep-file "${BLACKLIST_GREP_FILE}")
fi

# run aggregation
"${cmd[@]}"

# 2) compute KS p-values and EMD for each feature, variant 
# submit KS/EMD jobs via qsub
#bash analysis_tools/generate_metrics.sh "${WORKDIR}"
KS_DIR="${WORKDIR}/KS_EMD"
SCRIPTS_DIR="${KS_DIR}/scripts"
RESULTS_DIR="${KS_DIR}/results"
OUT_DIR="${WORKDIR}/metrics"

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
        cd $(pwd)
        echo "python3 analysis_tools/perform_KS_test.py $KS_DIR/${FILEPREFIX}.groupedvariants.piece${i}.pkl $KS_DIR/${FILEPREFIX}.WT.piece${i}.csv"
    ) > "$script_path"

    chmod u+x "$script_path"
    qsub -l mfree=20G "$script_path" # requires SGE cluster
done

echo "Waiting for KS/EMD qsub jobs to complete..."
while qstat -u "$USER" | grep -q "run_test_"; do # requires SGE cluster
  sleep 30
done
echo "All KS/EMD jobs have completed."

# verify that results landed in KS_EMD/results
# count pieces (PKL inputs)
num_pieces=$(ls "${KS_DIR}/${EXPERIMENT}.groupedvariants.piece"*.pkl | wc -l)
# each piece should produce 4 files: KS, loc, p, and EMD
expected=$(( num_pieces * 4 ))
# count found result files
found=$(ls "$RESULTS_DIR" | wc -l)
if [ ! -d "$RESULTS_DIR" ] || [ "$found" -lt "$expected" ]; then
  echo "Error: Expected at least $expected result files in $RESULTS_DIR but found $found."
  exit 1
fi
echo "Verified $found/$expected result files."

# 3) merge metrics + normalize + select
python3 analysis_tools/generate_profiles.py \
  --experiment          "${EXPERIMENT}" \
  --aggregated-median   "${WORKDIR}/${EXPERIMENT}_aggregated.parquet" \
  --ks-dir              "${KS_DIR}" \
  --metrics-dir         "${OUT_DIR}" \
  --blocklist-file      "${WORKDIR}/feature_blocklist.txt" \
  --nonblocked-file     "${WORKDIR}/nonblocked_features.txt" \
  --output-dir          "${WORKDIR}" \
  --normalize-to        "${NORMALIZE_TO}"
