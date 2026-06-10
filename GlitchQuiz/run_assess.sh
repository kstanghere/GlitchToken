#!/bin/bash

set -euo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ============ Configuration ============
MODEL_PATH="../../Models/Mistral/Mistral-7B-Instruct-v0.2"   # <-- Set your model path
NUM_GPUS=6                                                # Number of GPUs to use
MAX_NEW_TOKENS=128                                       # Max tokens to generate per response
SAVE_INTERVAL=100                                         # Save checkpoint every N tokens
MODEL_NAME="Mistral-7B-Instruct-v0.2"
# =======================================

RECORD_DIR="${BASE_DIR}/record/${MODEL_NAME}"
OUTPUT_DIR="${RECORD_DIR}/output"

mkdir -p "${OUTPUT_DIR}"

# Determine vocab size
VOCAB_SIZE=$(python3 - <<PY
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("${MODEL_PATH}", trust_remote_code=True)
print(len(tok))
PY
)

echo "========================================"
echo "Model: ${MODEL_NAME}"
echo "Path: ${MODEL_PATH}"
echo "Vocab size: ${VOCAB_SIZE}"
echo "GPUs: ${NUM_GPUS}"
echo "Output: ${OUTPUT_DIR}"
echo "========================================"

CHUNK_SIZE=$(( (VOCAB_SIZE + NUM_GPUS - 1) / NUM_GPUS ))
PIDS=()

for ((gpu=0; gpu<NUM_GPUS; gpu++)); do
    START=$(( gpu * CHUNK_SIZE ))
    END=$(( START + CHUNK_SIZE - 1 ))
    if (( START >= VOCAB_SIZE )); then
        echo "GPU ${gpu}: skip (start ${START} >= vocab ${VOCAB_SIZE})"
        continue
    fi
    if (( END >= VOCAB_SIZE )); then
        END=$(( VOCAB_SIZE - 1 ))
    fi

    LOG_FILE="${OUTPUT_DIR}/gpu${gpu}_token${START}-${END}.log"
    echo "GPU ${gpu}: tokens ${START}-${END} -> ${LOG_FILE}"

    CUDA_VISIBLE_DEVICES=${gpu} python3 "${BASE_DIR}/assess_local.py" \
        --model_path "${MODEL_PATH}" \
        --gpu_id 0 \
        --tokenID_start "${START}" \
        --tokenID_end "${END}" \
        --max_new_tokens "${MAX_NEW_TOKENS}" \
        --save_interval "${SAVE_INTERVAL}" \
        --output_dir "${OUTPUT_DIR}" \
        --model_name "${MODEL_NAME}" \
        --record_dir "${RECORD_DIR}" \
        --total_tokens "${VOCAB_SIZE}" \
        > "${LOG_FILE}" 2>&1 &

    PIDS+=($!)
    sleep 2
done

echo "Waiting for ${#PIDS[@]} workers..."
for pid in "${PIDS[@]}"; do
    wait "${pid}" || {
        echo "ERROR: worker pid ${pid} failed"
        exit 1
    }
done
echo "All workers finished."

echo "========================================"
echo "Running post-processing..."
echo "========================================"

python3 "${BASE_DIR}/post_process.py" \
    --model_name "${MODEL_NAME}" \
    --record_dir "${RECORD_DIR}" \
    --output_subdir "output" \
    --total_tokens "${VOCAB_SIZE}"

echo "All done. Results in: ${RECORD_DIR}"
