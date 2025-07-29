#!/bin/bash

PYTHON_BIN=$(which python)
USE_NOHUP=false

# Defaults
START=0
END=30
STEP=10

# Parse arguments
POSITIONAL=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --nohup)
            USE_NOHUP=true
            shift
            ;;
        *)
            POSITIONAL+=("$1")
            shift
            ;;
    esac
done

# Restore positional args
set -- "${POSITIONAL[@]}"

# Assign range values if provided
if [[ $# -ge 1 ]]; then START=$1; fi
if [[ $# -ge 2 ]]; then END=$2; fi
if [[ $# -ge 3 ]]; then STEP=$3; fi

# Validate input
if (( STEP <= 0 )); then
    echo "Error: STEP must be a positive integer." >&2
    exit 1
fi

if (( (END - START) % STEP != 0 )); then
    echo "Error: (END - START) must be divisible by STEP." >&2
    exit 1
fi

echo "Using Python binary: $PYTHON_BIN"
echo "Using nohup: $USE_NOHUP"
echo "Range: $START to $END with step $STEP"

# Launch loop
for ((i = START; i < END; i += STEP)); do
    j=$((i + STEP))
    name="seeds_${i}_${j}"
    echo "Launching $name"
    if $USE_NOHUP; then
        nohup "$PYTHON_BIN" run_seeds.py $i $j > logs/$name.log 2>&1 &
    else
        "$PYTHON_BIN" run_seeds.py $i $j > logs/$name.log 2>&1 &
    fi
done

wait
