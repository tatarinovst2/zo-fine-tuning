#!/bin/bash

set -euo pipefail

CHECKPOINTS_DIR=""

FORWARD_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --checkpoints_dir)
            CHECKPOINTS_DIR="$2"
            shift 2
            ;;
        --output_path|--output_dir)
            echo "Do not pass --output_path or --output_dir manually."
            exit 1
            ;;
        *)
            FORWARD_ARGS+=("$1")
            shift

            if [[ $# -gt 0 && ! "$1" =~ ^-- ]]; then
                FORWARD_ARGS+=("$1")
                shift
            fi
            ;;
    esac
done

if [[ -z "$CHECKPOINTS_DIR" ]]; then
    echo "Missing --checkpoints_dir"
    exit 1
fi

OUTPUT_DIR="$CHECKPOINTS_DIR"

# Extract dataset name from --data_file

DATA_FILE=""

for ((i=0; i<${#FORWARD_ARGS[@]}; i++)); do
    if [[ "${FORWARD_ARGS[$i]}" == "--data_file" ]]; then
        DATA_FILE="${FORWARD_ARGS[$((i+1))]}"
        break
    fi
done

if [[ -z "$DATA_FILE" ]]; then
    echo "Missing --data_file"
    exit 1
fi

DATASET_NAME="$(basename "$(dirname "$DATA_FILE")")"
RUN_NAME="$(basename "$CHECKPOINTS_DIR")"

echo "Dataset: $DATASET_NAME"
echo "Run name: $RUN_NAME"
echo "Output dir: $OUTPUT_DIR"

# Iterate over checkpoints

for CKPT_PATH in "$CHECKPOINTS_DIR"/*; do
    if [[ ! -d "$CKPT_PATH" ]]; then
        continue
    fi

    CKPT_NAME="$(basename "$CKPT_PATH")"

    OUTPUT_PATH="${OUTPUT_DIR}/${RUN_NAME}__${CKPT_NAME}__${DATASET_NAME}.jsonl"

    echo "======================================================"
    echo "Checkpoint: $CKPT_NAME"
    echo "Output:     $OUTPUT_PATH"
    echo "======================================================"

    python pipeline/inference.py \
        --model_name_or_path "$CKPT_PATH" \
        --output_path "$OUTPUT_PATH" \
        "${FORWARD_ARGS[@]}"
done

echo "Done."
