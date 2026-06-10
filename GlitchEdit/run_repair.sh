#!/bin/bash
set -euo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ============ Configuration ============
MODEL_PATH="../../Models/Mistral/Mistral-7B-Instruct-v0.2" # <-- Set your model path
GPU_ID=6                                                 # GPU to use
MAX_NEW_TOKENS=128                                       # Max tokens per generation
SAVE_EVERY=10                                            # Save report every N tokens
# =======================================

MODE="${1:-tiny}"
MAX_TOKENS="${2:-}"

if [ "$MODE" = "full" ]; then
    GLITCH_FILE="${BASE_DIR}/glitch_tokens_full.json"
    OUTPUT_DIR="${BASE_DIR}/output/full"
else
    GLITCH_FILE="${BASE_DIR}/glitch_tokens_tiny.json"
    OUTPUT_DIR="${BASE_DIR}/output/tiny"
fi

mkdir -p "${OUTPUT_DIR}"

CMD="CUDA_VISIBLE_DEVICES=${GPU_ID} python3 ${BASE_DIR}/glitch_repair.py \
    --model_path ${MODEL_PATH} \
    --glitch_tokens ${GLITCH_FILE} \
    --tasks ${BASE_DIR}/task_8_tasks.json \
    --output_path ${OUTPUT_DIR} \
    --max_new_tokens ${MAX_NEW_TOKENS} \
    --save_every ${SAVE_EVERY}"

if [ -n "${MAX_TOKENS}" ]; then
    CMD="${CMD} --max_tokens ${MAX_TOKENS}"
fi

echo "========================================"
echo "GlitchEdit Repair"
echo "Mode: ${MODE}"
echo "Glitch tokens: ${GLITCH_FILE}"
echo "Output: ${OUTPUT_DIR}"
echo "GPU: ${GPU_ID}"
echo "Save every: ${SAVE_EVERY} tokens"
if [ -n "${MAX_TOKENS}" ]; then
    echo "Max tokens to repair: ${MAX_TOKENS}"
fi
echo "========================================"

LOG_FILE="${OUTPUT_DIR}/repair.log"
echo "Logging to: ${LOG_FILE}"

eval ${CMD} 2>&1 | tee "${LOG_FILE}"
