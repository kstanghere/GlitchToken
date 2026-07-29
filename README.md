# GlitchToken: Glitch Token Detection and Repair

This repository provides tools for detecting and repairing glitch tokens in Large Language Models (LLMs). Glitch tokens are tokens that cause unexpected or erroneous behavior in LLM outputs, including failure to repeat, count, spell, or construct sentences with the token.

We demonstrate our approach using Mistral-7B-Instruct-v0.2 as the evaluation model.

---

## Repository Structure

```
GlitchToken/
├── README.md                              # This file
├── requirements.txt                       # Python dependencies (excluding PyTorch)
├── GlitchQuiz/                            # Glitch Token Detection Module
│   ├── assess_local.py                    # Per-GPU detection worker
│   ├── post_process.py                    # Merge shard outputs and compute stats
│   ├── judge_token.py                     # Glitch task judging logic
│   ├── run_assess.sh                      # Multi-GPU launch script
│   └── task_template_en1.json             # 8 glitch detection task templates
└── GlitchEdit/                            # Glitch Token Repair Module
    ├── glitch_repair.py                   # Embedding repair script
    ├── run_repair.sh                      # Repair launch script
    ├── task_8_tasks.json                  # 8 glitch task templates
    └── glitch_tokens_tiny.json            # 400-token sample for evaluation
```

---

## Environment Setup

### Step 1: Create Conda Environment

```bash
conda create -n glitchtoken python=3.10 -y
conda activate glitchtoken
```

### Step 2: Install PyTorch 2.9.0

Install PyTorch with CUDA 12.8 support:

```bash
pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 --index-url https://download.pytorch.org/whl/cu128
```

### Step 3: Verify PyTorch Installation

```bash
python -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

Ensure the output shows `CUDA available: True`. If not, check your CUDA driver and PyTorch installation.

### Step 4: Install Other Dependencies

```bash
cd ./GlitchToken
pip install -r requirements.txt
```

---

## Model Weights Download

Download Mistral-7B-Instruct-v0.2 from Hugging Face:

```bash
# Option 1: git clone (requires git-lfs)
git lfs install
git clone https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2 ./models/Mistral-7B-Instruct-v0.2

# Option 2: huggingface-cli
huggingface-cli download mistralai/Mistral-7B-Instruct-v0.2 --local-dir ./models/Mistral-7B-Instruct-v0.2
```

For more information: [https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2)

---

## GlitchQuiz: Glitch Token Detection

GlitchQuiz evaluates every token in the model's vocabulary across 8 tasks to identify glitch tokens. The output contains only detection results (identified glitch tokens and summary statistics).

### Usage

Edit `run_assess.sh` to set your model path and the number of GPUs:

```bash
MODEL_PATH="/path/to/Mistral-7B-Instruct-v0.2"   # <-- Change this
NUM_GPUS=6                                         # Number of GPUs to use (set to 1 for single GPU)
```

Then run:

```bash
cd GlitchQuiz
chmod +x run_assess.sh
./run_assess.sh
```

The script automatically:

1. Detects the vocabulary size (32000 for Mistral)
2. Splits the vocab evenly across GPUs
3. Launches parallel workers
4. Runs post-processing to merge results

### Output Files

After completion, you will find in `record/Mistral-7B-Instruct-v0.2/`:

| File | Description |
| --- | --- |
| `Mistral-7B-Instruct-v0.2_glitch_tokens.json` | List of detected glitch tokens |
| `Mistral-7B-Instruct-v0.2_detection_stats.json` | Detection statistics |
| `output/gpu*_token*.log` | Per-GPU execution logs |

Example `Mistral-7B-Instruct-v0.2_glitch_tokens.json`:

```json
[
  {
    "tokenID": "1234",
    "tokenDecode": "▁example",
    "wrong_tasks": "1,3,5"
  }
]
```

- `tokenID`: Token ID in the vocabulary
- `tokenDecode`: Decoded string of the token
- `wrong_tasks`: Comma-separated IDs of failed glitch tasks (1–8)

Example `Mistral-7B-Instruct-v0.2_detection_stats.json`:

```json
{
  "total_glitch_tokens": 1523,
  "total_vocab_tokens": 32000,
  "glitch_token_ratio": 4.76
}
```

---

## GlitchEdit: Glitch Token Repair

GlitchEdit repairs glitch tokens by modifying their embeddings through interpolation with similar normal tokens. It performs a grid search over interpolation parameters (alpha, k) to find the optimal repair. The output contains only repair results and fix rate metrics.

To balance evaluation time and ethical considerations, we provide a **tiny set** (`glitch_tokens_tiny.json`) containing 400 randomly sampled glitch tokens for testing. 

### Usage

Edit `run_repair.sh` to set your model path, then run:

```bash
cd GlitchEdit
chmod +x run_repair.sh
./run_repair.sh
```

This repairs the tiny set (400 tokens, approximately 2-3 hours on a single GPU).

You can also limit the number of tokens to repair:

```bash
./run_repair.sh tiny 100
```

### Output

After completion, `output/tiny/` contains:

- `repair_report.json` — Repair results and metrics for each token
- `repaired_model_final/` — The repaired model weights (saved via `model.save_pretrained()`)

Example `repair_report.json`:

```json
{
  "summary": {
    "total_tokens_processed": 400,
    "successful_fixed": 276,
    "partial_fixed": 63,
    "fallback_fixed": 15,
    "failed_fixes": 46,
    "skipped_tokens": 0,
    "error_tokens": 0,
    "fix_rate": 88.5
  },
  "token_results": [
    {
      "tokenID": "1234",
      "original_wrong_tasks": "1,2,3",
      "repair_status": "success",
      "repair_details": {"alpha": 0.3, "k": 2},
      "fixed_tasks": ["task1", "task2", "task3"],
      "remaining_wrong_tasks": []
    }
  ]
}
```

Summary fields:

- `successful_fixed`: All wrong tasks fixed via grid search
- `partial_fixed`: Some wrong tasks fixed via grid search
- `fallback_fixed`: All wrong tasks fixed via fallback strategy (alpha=1.0)
- `failed_fixes`: Repair failed
- `skipped_tokens`: Tokens with empty `wrong_tasks` field
- `error_tokens`: Tokens that raised an exception during repair
- `fix_rate`: Percentage of tokens with status `success`, `partial`, or `fallback`

Per-token `repair_status` values: `success`, `partial`, `fallback`, `failed`, `skipped`, `error`.

---

## Adapting to Other Models

While this artifact uses Mistral-7B-Instruct-v0.2 as the demonstration model, the approach can be adapted to other models:

1. **Chat Template**: Add `--use_hf_chat_template` flag to use the model's built-in chat template via `tokenizer.apply_chat_template()`. This works for most HuggingFace models that define a chat template in their tokenizer config.
2. **Model Loading**: The scripts use `AutoModelForCausalLM` and `AutoTokenizer`, which are compatible with most causal LM architectures.
3. **Vocabulary Size**: The scripts automatically detect vocab size from the tokenizer.

Example for a different model:

```bash
# GlitchQuiz with a different model
CUDA_VISIBLE_DEVICES=0 python assess_local.py \
    --model_path /path/to/other-model \
    --gpu_id 0 \
    --tokenID_start 0 --tokenID_end 31999 \
    --max_new_tokens 128 \
    --output_dir ./record/other-model/output \
    --model_name other-model \
    --use_hf_chat_template

# GlitchEdit with a different model
python glitch_repair.py \
    --model_path /path/to/other-model \
    --glitch_tokens ./detected_glitch_tokens.json \
    --tasks ./task_8_tasks.json \
    --output_path ./output \
    --use_hf_chat_template
```

---

## Ethical Considerations

This research is intended to improve the safety and robustness of Large Language Models by identifying and repairing glitch tokens. We acknowledge the dual-use potential of glitch token detection tools. To mitigate risks:

1. The provided `glitch_tokens_tiny.json` is a small randomly sampled subset intended solely for artifact evaluation purposes.
2. This repository is prohibited from being used for any malicious purposes, including but not limited to exploiting glitch tokens to induce harmful model behaviors.

We encourage responsible use of this research to advance LLM safety.
