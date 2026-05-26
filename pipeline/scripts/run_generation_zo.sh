#!/bin/bash

set -e

MODEL_NAME_OR_PATH="Qwen/Qwen2.5-0.5B"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  key="$1"

  case $key in
    --dataset_dir)
      DATASET_DIR="$2"
      shift 2
      ;;

    --model_name_or_path)
      MODEL_NAME_OR_PATH="$2"
      shift 2
      ;;

    --learning_rate)
      LR="$2"
      shift 2
      ;;

    --trainer_type)
      TRAINER_TYPE="$2"
      shift 2
      ;;

    *)
      EXTRA_ARGS+=("$1")  # Preserve all unknown args

      # If next token exists and is not another flag,
      # treat it as the value for this arg.
      if [[ $# -gt 1 && ! "$2" =~ ^-- ]]; then
        EXTRA_ARGS+=("$2")
        shift
      fi

      shift
      ;;
  esac
done

if [[ -z "$DATASET_DIR" || -z "$TRAINER_TYPE" || -z "$LR" ]]; then
  echo "Usage:"
  echo "$0 \\"
  echo "  --dataset_dir <dataset> \\"
  echo "  --trainer_type <trainer> \\"
  echo "  --learning_rate <lr> \\"
  echo "  [--model_name_or_path] <model> \\"
  echo "  [--per_device_train_batch_size <bs>] \\"
  echo "  [extra args ...]"
  exit 1
fi

# Build experiment/checkpoint name
sanitize() {
  echo "$1" | sed 's#[/ ]#-#g'
}
basename_sanitized() {
  local p="${1%/}"      # strip trailing /
  echo "$(basename "$p")" | sed 's#[/ ]#-#g'
}

EXP_NAME="${TRAINER_TYPE}"
EXP_NAME+="_$(basename_sanitized "$DATASET_DIR")"
EXP_NAME+="_lr$(sanitize "$LR")"

# Include extra args in experiment name
for ((i=0; i<${#EXTRA_ARGS[@]}; i++)); do
  arg="${EXTRA_ARGS[$i]}"

  if [[ "$arg" == --* ]]; then
    key=$(echo "$arg" | sed 's/^--//')

    next_index=$((i + 1))

    if [[ $next_index -lt ${#EXTRA_ARGS[@]} ]]; then
      value="${EXTRA_ARGS[$next_index]}"

      if [[ "$value" != --* ]]; then
        EXP_NAME+="_${key}$(sanitize "$value")"
      else
        EXP_NAME+="_${key}"
      fi
    else
      EXP_NAME+="_${key}"
    fi
  fi
done

OUTPUT_DIR="./checkpoints/${EXP_NAME}"

echo "Experiment name: $EXP_NAME"
echo "Output dir: $OUTPUT_DIR"

MAX_STEPS=20000
SAVE_STEPS=2000
EVAL_STEPS=100
LOGGING_STEPS=100

if [[ "$TRAINER_TYPE" == "zomuon" || "$TRAINER_TYPE" == "zomuonm" ]]; then
  MAX_STEPS=8000
  SAVE_STEPS=800
  EVAL_STEPS=40
  LOGGING_STEPS=40
fi

if [[ "$TRAINER_TYPE" == "hizoo" ]]; then
  MAX_STEPS=13200
  SAVE_STEPS=1320
  EVAL_STEPS=66
  LOGGING_STEPS=66
fi

if [[ "$TRAINER_TYPE" == "fzoo" ]]; then
  MAX_STEPS=4400
  SAVE_STEPS=440
  EVAL_STEPS=22
  LOGGING_STEPS=22
fi

CMD=(
  python pipeline/run_experiment.py
  --model_name_or_path "$MODEL_NAME_OR_PATH"
  --dataset_dir "$DATASET_DIR"
  --output_dir "$OUTPUT_DIR"
  --task_type "generation"
  --max_steps "$MAX_STEPS"
  --trainer_type "$TRAINER_TYPE"
  --per_device_train_batch_size 16
  --per_device_eval_batch_size 16
  --gradient_accumulation_steps 1
  --evaluation_strategy "steps"
  --eval_steps "$EVAL_STEPS"
  --logging_steps "$LOGGING_STEPS"
  --save_strategy "steps"
  --save_steps "$SAVE_STEPS"
  --save_total_limit 21
  --learning_rate "$LR"
  --weight_decay 0.0
  --warmup_ratio 0.00
  --lr_scheduler_type "constant"
  --gradient_checkpointing False
  --report_to none
  --save_only_model True
  "${EXTRA_ARGS[@]}"
)

echo "Running command:"
printf '%q ' "${CMD[@]}"
echo

"${CMD[@]}"
