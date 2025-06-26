#!/bin/bash

WORKINGDIR=$(pwd)
UTESTDIR="$1"

if [[ -z "$UTESTDIR" ]]; then
    echo "Usage: $0 <directory>"
    exit 1
fi

# Ensure the target directory exists
if [[ ! -d "$UTESTDIR" ]]; then
    echo "Error: Directory $UTESTDIR does not exist."
    exit 1
fi

# Get the file prefix from the first matching file
FILEPREFIX=""
for file in "${UTESTDIR}"/*.piece0.pkl; do
    FILEPREFIX=$(basename "$file" | cut -d '.' -f 1)  # Extract prefix
    break  # Exit after the first match
done
if [[ -z "$FILEPREFIX" ]]; then
    echo "No files matching '*.piece0.pkl' found in $UTESTDIR."
    exit 1
fi

# Create directories if they do not exist
mkdir -p "${UTESTDIR}/results"
mkdir -p "${UTESTDIR}/scripts"

# Count files for nparts
nparts=$(($(ls -1 "${UTESTDIR}"/*.pkl 2>/dev/null | wc -l) - 1))
if [[ "$nparts" -lt 0 ]]; then
    echo "No .pkl files found in $UTESTDIR."
    exit 1
fi
echo "Found $((nparts + 1)) files."

# Loop through parts and create scripts
for i in $(seq 0 "$nparts"); do
    script_path="${UTESTDIR}/scripts/run_u_test_${i}.sh"
    (
        echo "#!/bin/bash"
        echo "source activate pycytominer"
        echo "cd ${WORKINGDIR}"
        echo "python perform_KS_test.py ${UTESTDIR}/${FILEPREFIX}.groupedvariants.piece${i}.pkl ${UTESTDIR}/${FILEPREFIX}.WT.piece${i}.csv"
    ) > "$script_path"

    chmod u+x "$script_path"
    qsub -l h=fl004 -l mfree=20G "$script_path"
done
