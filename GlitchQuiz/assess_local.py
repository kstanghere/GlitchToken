#!/usr/bin/env python3
import argparse
import json
import os
import re
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from judge_token import is_incorrect

DEFAULT_SYSTEM_PROMPT = "You are a very helpful AI bot that can reply to the question from users."

GLITCH_TASK_NAMES = [f"task{i}" for i in range(1, 9)]

MISTRAL_TEMPLATES = {
    "single_round": "<s>{system}\n\n[INST]{prompt}[/INST]",
    "multi_round": "<s>{system}\n\n[INST]{prompt1}[/INST]{answer1}</s>[INST]{prompt2}[/INST]",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Local glitch token assessment")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--gpu_id", type=int, required=True)
    parser.add_argument("--tokenID_start", type=int, required=True)
    parser.add_argument("--tokenID_end", type=int, required=True)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Temperature for generation (only effective when do_sample=True)")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model_name", type=str, default=None,
                        help="Model name for output files (default: basename of model_path)")
    parser.add_argument("--record_dir", type=str, default=None,
                        help="Parent dir where merged files are written (default: parent of output_dir)")
    parser.add_argument("--total_tokens", type=int, default=None,
                        help="Total vocab size for computing glitch_token_ratio")
    parser.add_argument("--glitch_template_file", type=str, default=None,
                        help="JSON with task templates (default: task_template_en1.json next to script)")
    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument("--use_hf_chat_template", action="store_true",
                        help="Use tokenizer.apply_chat_template() instead of manual Mistral template")
    return parser.parse_args()


def load_templates(path: str, task_names: list) -> list:
    with open(path, "r", encoding="utf-8") as f:
        all_tasks = json.load(f)
    name_set = set(task_names)
    tasks = [t for t in all_tasks if t["task"] in name_set]
    if len(tasks) != len(task_names):
        found = {t["task"] for t in tasks}
        missing = name_set - found
        raise ValueError(f"Missing tasks in {path}: {missing}")
    order = {name: i for i, name in enumerate(task_names)}
    tasks.sort(key=lambda t: order[t["task"]])
    return tasks


def load_model_and_tokenizer(model_path: str, gpu_id: int):
    device = f"cuda:{gpu_id}"
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map={"": device},
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer, device


def format_prompt(tokenizer, prompt: str, use_hf_template: bool, system: str = DEFAULT_SYSTEM_PROMPT) -> str:
    if use_hf_template:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except TypeError:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        return MISTRAL_TEMPLATES["single_round"].format(system=system, prompt=prompt)


def format_multi_round(tokenizer, prompt1: str, answer1: str, prompt2: str,
                       use_hf_template: bool, system: str = DEFAULT_SYSTEM_PROMPT) -> str:
    if use_hf_template:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt1},
            {"role": "assistant", "content": answer1},
            {"role": "user", "content": prompt2},
        ]
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except TypeError:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        return MISTRAL_TEMPLATES["multi_round"].format(
            system=system, prompt1=prompt1, answer1=answer1, prompt2=prompt2)


def generate_response(model, tokenizer, device, formatted_prompt: str, max_new_tokens: int) -> str:
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)
    input_len = inputs["input_ids"].shape[-1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
    return response


def run_glitch_tasks(model, tokenizer, device, token_decode, glitch_tasks,
                     max_new_tokens, use_hf_template):
    task_results = {}
    for task in glitch_tasks:
        task_name = task["task"]
        if task.get("muilti_round") == "True":
            prompt1 = task["template1"].replace("{token}", token_decode)
            formatted1 = format_prompt(tokenizer, prompt1, use_hf_template)
            answer1 = generate_response(model, tokenizer, device, formatted1, max(1, max_new_tokens // 2))

            prompt2 = task["template2"].replace("{token}", token_decode)
            formatted2 = format_multi_round(tokenizer, prompt1, answer1, prompt2, use_hf_template)
            answer2 = generate_response(model, tokenizer, device, formatted2, max_new_tokens)

            task_results[task_name] = {
                "task_type": task["type"],
                "muilti_round": "True",
                "prompt1": prompt1,
                "answer1": answer1,
                "prompt2": prompt2,
                "answer2": answer2,
            }
        else:
            prompt = task["template1"].replace("{token}", token_decode)
            formatted = format_prompt(tokenizer, prompt, use_hf_template)
            answer = generate_response(model, tokenizer, device, formatted, max_new_tokens)
            task_results[task_name] = {
                "task_type": task["type"],
                "muilti_round": "False",
                "prompt1": prompt,
                "answer1": answer,
            }
    return task_results


def judge_glitch_token(token_decode: str, task_results: dict, model_name: str) -> list:
    wrong_tasks = []
    for task_num in range(1, 9):
        task_name = f"task{task_num}"
        if task_name not in task_results:
            continue
        tr = task_results[task_name]
        answer1 = tr.get("answer1", "")
        answer2 = tr.get("answer2") if task_num == 8 else None
        if is_incorrect(task_name, token_decode, answer1, answer2, model_name):
            wrong_tasks.append(task_num)
    return wrong_tasks


def save_outputs(output_dir, token_start, token_end, glitch_tokens):
    os.makedirs(output_dir, exist_ok=True)
    suffix = f"{token_start}-{token_end}"
    path = os.path.join(output_dir, f"glitch_tokens_{suffix}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(glitch_tokens, f, ensure_ascii=False, indent=2)
    return path


def main():
    args = parse_args()
    script_dir = os.path.dirname(os.path.abspath(__file__))

    glitch_template_file = args.glitch_template_file or os.path.join(
        script_dir, "task_template_en1.json")

    glitch_tasks = load_templates(glitch_template_file, GLITCH_TASK_NAMES)

    os.makedirs(args.output_dir, exist_ok=True)
    model_name = args.model_name or os.path.basename(args.model_path.rstrip("/"))
    record_dir = args.record_dir or os.path.dirname(args.output_dir.rstrip("/"))

    print(f"GPU {args.gpu_id}: loading model from {args.model_path}")
    model, tokenizer, device = load_model_and_tokenizer(args.model_path, args.gpu_id)
    vocab_size = len(tokenizer)

    token_end = min(args.tokenID_end, vocab_size - 1)
    suffix = f"{args.tokenID_start}-{token_end}"

    glitch_tokens = []
    resume_from = args.tokenID_start

    ckpt_glitch = os.path.join(args.output_dir, f"glitch_tokens_{suffix}.json")

    if os.path.isfile(ckpt_glitch):
        try:
            with open(ckpt_glitch, "r", encoding="utf-8") as f:
                glitch_tokens = json.load(f)
            if glitch_tokens:
                last_token_id = max(int(t["tokenID"]) for t in glitch_tokens)
                resume_from = last_token_id + 1
                print(f"GPU {args.gpu_id}: RESUMING from token {resume_from} "
                      f"(loaded {len(glitch_tokens)} glitch)",
                      flush=True)
        except Exception as e:
            print(f"GPU {args.gpu_id}: failed to load checkpoint, starting fresh: {e}", flush=True)
            glitch_tokens = []
            resume_from = args.tokenID_start

    total = token_end - args.tokenID_start + 1
    already_done = resume_from - args.tokenID_start

    if resume_from > token_end:
        print(f"GPU {args.gpu_id}: all {total} tokens already done, skipping.", flush=True)
        return

    print(f"GPU {args.gpu_id}: processing tokens {resume_from}-{token_end} "
          f"({total - already_done} remaining of {total} total), "
          f"max_new_tokens={args.max_new_tokens}", flush=True)

    import warnings
    import logging
    warnings.filterwarnings("ignore", message=".*pad_token_id.*")
    logging.getLogger("transformers.generation").setLevel(logging.ERROR)

    start_wall = time.time()

    for idx_offset, token_id in enumerate(range(resume_from, token_end + 1)):
        idx = already_done + idx_offset
        token_decode = tokenizer.decode([token_id])
        t0 = time.time()

        task_results = run_glitch_tasks(
            model, tokenizer, device, token_decode, glitch_tasks,
            args.max_new_tokens, args.use_hf_chat_template)

        wrong_tasks = judge_glitch_token(token_decode, task_results, model_name)

        is_glitch = False
        if wrong_tasks:
            is_glitch = True
            glitch_tokens.append({
                "tokenID": str(token_id),
                "tokenDecode": token_decode,
                "wrong_tasks": ",".join(map(str, sorted(wrong_tasks))),
            })

        token_time = time.time() - t0
        elapsed_total = time.time() - start_wall
        done = idx + 1
        new_done = idx_offset + 1
        remaining = total - done
        avg_time = elapsed_total / new_done
        eta_min = (avg_time * remaining) / 60
        glitch_tag = f" GLITCH wrong={wrong_tasks}" if is_glitch else ""
        print(f"[GPU {args.gpu_id}] {done}/{total} "
              f"token={token_id} ({token_time:.1f}s){glitch_tag} "
              f"| glitch_so_far={len(glitch_tokens)} "
              f"| ETA={eta_min:.1f}min", flush=True)

        if new_done % args.save_interval == 0 or token_id == token_end:
            save_outputs(args.output_dir, args.tokenID_start, token_end, glitch_tokens)

    save_outputs(args.output_dir, args.tokenID_start, token_end, glitch_tokens)
    print(f"GPU {args.gpu_id}: done. glitch_tokens={len(glitch_tokens)}")


if __name__ == "__main__":
    main()
