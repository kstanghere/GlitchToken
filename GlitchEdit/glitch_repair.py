#!/usr/bin/env python3
import os
import re
import json
import torch
import difflib
import argparse
import time
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

DEFAULT_SYSTEM_PROMPT = "You are a very helpful AI bot that can reply to the question from users."

MISTRAL_TEMPLATES = {
    "single_round": "<s>{system}\n\n[INST]{prompt}[/INST]",
    "multi_round": "<s>{system}\n\n[INST]{prompt1}[/INST]{answer1}</s>[INST]{prompt2}[/INST]",
}

def contains_non_ascii(text):
    return any(ord(char) > 127 for char in text)


def get_ascii_representation(text):
    if not text:
        return ""
    return json.dumps(text, ensure_ascii=True)[1:-1]


def clean_token(token_decode_lower):
    chars_to_strip = ' .,;:!?-_\'"\n\t\r<>[]{}()\\|`~@#$%^&*+='
    return ''.join(char for char in token_decode_lower if char not in chars_to_strip)


def num_to_word(num):
    ones = ["", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
    teens = ["ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
             "sixteen", "seventeen", "eighteen", "nineteen"]
    tens = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
    if num == 0:
        return "zero"
    elif 1 <= num <= 9:
        return ones[num]
    elif 10 <= num <= 19:
        return teens[num - 10]
    elif 20 <= num <= 99:
        return tens[num // 10] + ("-" + ones[num % 10] if num % 10 else "")
    elif num == 100:
        return "one hundred"
    return ""


def is_task_correct(task_name, token_decode, answer):
    has_non_ascii_token = contains_non_ascii(token_decode) if token_decode else False
    has_non_ascii_answer = contains_non_ascii(answer) if answer else False

    token_ascii = get_ascii_representation(token_decode) if has_non_ascii_token else token_decode
    answer_ascii = get_ascii_representation(answer) if has_non_ascii_answer else answer

    token_decode_lower = token_decode.lower() if token_decode else ""
    token_ascii_lower = token_ascii.lower() if token_ascii else ""
    answer_lower = answer.lower() if answer else ""
    answer_ascii_lower = answer_ascii.lower() if answer_ascii else ""
    cleaned_token_lower = clean_token(token_decode_lower)
    cleaned_token_ascii_lower = clean_token(token_ascii_lower)

    def token_exists_in_answer(t_text, t_ascii, a_text, a_ascii):
        return (t_text in a_text or t_text in a_ascii or t_ascii in a_text or t_ascii in a_ascii)

    if task_name == "task1":
        has_correct_number = any(
            s in answer_lower or s in answer_ascii_lower for s in ["6", "six"])
        token_exists = token_exists_in_answer(
            token_decode_lower, token_ascii_lower, answer_lower, answer_ascii_lower)
        cleaned_exists = (cleaned_token_lower in answer_lower or
                          cleaned_token_lower in answer_ascii_lower or
                          cleaned_token_ascii_lower in answer_lower or
                          cleaned_token_ascii_lower in answer_ascii_lower)
        return has_correct_number or token_exists or (cleaned_token_lower != "" and cleaned_exists)

    elif task_name == "task2":
        if has_non_ascii_token:
            char_count = len(re.findall(r'\\u[0-9a-f]{4}', token_ascii_lower))
        else:
            char_count = len(token_decode_lower)
        cleaned_char_count = len(cleaned_token_lower)
        count_mentioned = (str(char_count) in answer_lower or
                           str(char_count) in answer_ascii_lower or
                           num_to_word(char_count) in answer_lower or
                           num_to_word(char_count) in answer_ascii_lower or
                           (cleaned_char_count != char_count and (
                               str(cleaned_char_count) in answer_lower or
                               str(cleaned_char_count) in answer_ascii_lower or
                               num_to_word(cleaned_char_count) in answer_lower or
                               num_to_word(cleaned_char_count) in answer_ascii_lower)))
        token_exists = token_exists_in_answer(
            token_decode_lower, token_ascii_lower, answer_lower, answer_ascii_lower)
        cleaned_exists = (cleaned_token_lower in answer_lower or
                          cleaned_token_lower in answer_ascii_lower or
                          cleaned_token_ascii_lower in answer_lower or
                          cleaned_token_ascii_lower in answer_ascii_lower)
        return count_mentioned or token_exists or (cleaned_token_lower != "" and cleaned_exists)

    elif task_name in ["task3", "task4"]:
        token_exists = token_exists_in_answer(
            token_decode_lower, token_ascii_lower, answer_lower, answer_ascii_lower)
        cleaned_exists = (cleaned_token_lower in answer_lower or
                          cleaned_token_lower in answer_ascii_lower or
                          cleaned_token_ascii_lower in answer_lower or
                          cleaned_token_ascii_lower in answer_ascii_lower)
        return token_exists or (cleaned_token_lower != "" and cleaned_exists)

    elif task_name == "task5":
        original_count = answer_lower.count(token_decode_lower) + answer_ascii_lower.count(token_decode_lower)
        ascii_count = answer_lower.count(token_ascii_lower) + answer_ascii_lower.count(token_ascii_lower)
        cleaned_count = answer_lower.count(cleaned_token_lower) + answer_ascii_lower.count(cleaned_token_lower)
        cleaned_ascii_count = (answer_lower.count(cleaned_token_ascii_lower) +
                               answer_ascii_lower.count(cleaned_token_ascii_lower))
        return (original_count >= 2 or ascii_count >= 2 or
                (cleaned_token_lower != "" and (cleaned_count >= 2 or cleaned_ascii_count >= 2)))

    elif task_name == "task6":
        token_exists = token_exists_in_answer(
            token_decode_lower, token_ascii_lower, answer_lower, answer_ascii_lower)
        cleaned_exists = (cleaned_token_lower in answer_lower or
                          cleaned_token_lower in answer_ascii_lower or
                          cleaned_token_ascii_lower in answer_lower or
                          cleaned_token_ascii_lower in answer_ascii_lower)
        return token_exists or (cleaned_token_lower != "" and cleaned_exists)

    elif task_name == "task7":
        if has_non_ascii_token:
            chars = re.findall(r'\\u[0-9a-f]{4}|.', token_ascii_lower)
            hyphenated = '-'.join(chars)
            chars_in_token = set(chars)
        else:
            hyphenated = '-'.join(list(token_decode_lower))
            chars_in_token = set(cleaned_token_lower)
        all_chars_present = all(char in answer_lower or char in answer_ascii_lower for char in chars_in_token)
        hyphenated_exists = hyphenated in answer_lower or hyphenated in answer_ascii_lower
        token_exists = token_exists_in_answer(
            token_decode_lower, token_ascii_lower, answer_lower, answer_ascii_lower)
        cleaned_exists = (cleaned_token_lower in answer_lower or
                          cleaned_token_lower in answer_ascii_lower or
                          cleaned_token_ascii_lower in answer_lower or
                          cleaned_token_ascii_lower in answer_ascii_lower)
        return (hyphenated_exists or token_exists or
                (cleaned_token_lower != "" and (cleaned_exists or all_chars_present)))

    elif task_name == "task8":
        token_exists = token_exists_in_answer(
            token_decode_lower, token_ascii_lower, answer_lower, answer_ascii_lower)
        cleaned_exists = (cleaned_token_lower in answer_lower or
                          cleaned_token_lower in answer_ascii_lower or
                          cleaned_token_ascii_lower in answer_lower or
                          cleaned_token_ascii_lower in answer_ascii_lower)
        return token_exists or (cleaned_token_lower != "" and cleaned_exists)

    return False


def load_model_and_tokenizer(model_path):
    print(f"Loading model and tokenizer from {model_path}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, trust_remote_code=True).to(device)
    model.eval()
    return model, tokenizer


def find_similar_normal_tokens(glitch_token_decode, tokenizer, top_k=10):
    all_tokens = tokenizer.get_vocab()
    normal_tokens = {token: idx for token, idx in all_tokens.items()
                     if token != glitch_token_decode}
    similarities = []
    for token in normal_tokens.keys():
        similarity = difflib.SequenceMatcher(None, glitch_token_decode, token).ratio()
        similarities.append((token, similarity))
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_k]


def create_modified_embedding(model, glitch_token_id, similar_tokens, alpha, k, tokenizer):
    embeddings = model.get_input_embeddings()
    original_embedding = embeddings.weight[glitch_token_id].clone()

    if k == 0 or alpha == 0:
        return original_embedding

    top_k_tokens = similar_tokens[:k]
    total_similarity = sum(sim for _, sim in top_k_tokens)
    if total_similarity == 0:
        weights = [1 / k] * k
    else:
        weights = [sim / total_similarity for _, sim in top_k_tokens]

    normal_embedding = torch.zeros_like(original_embedding)
    for i, (token, _) in enumerate(top_k_tokens):
        token_id = tokenizer.convert_tokens_to_ids(token)
        normal_embedding += weights[i] * embeddings.weight[token_id]

    modified_embedding = alpha * normal_embedding + (1 - alpha) * original_embedding
    modified_embedding = torch.nn.functional.normalize(modified_embedding, p=2, dim=-1)
    return modified_embedding


def apply_embedding_modification(model, token_id, modified_embedding):
    with torch.no_grad():
        embeddings = model.get_input_embeddings()
        embeddings.weight[token_id] = modified_embedding


def generate_task_response(model, tokenizer, template, token_decode, max_new_tokens,
                           use_hf_template=False):
    prompt = template.replace("{token}", token_decode)
    if use_hf_template:
        messages = [
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        formatted_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
    else:
        formatted_prompt = MISTRAL_TEMPLATES["single_round"].format(
            system=DEFAULT_SYSTEM_PROMPT, prompt=prompt)

    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    input_len = inputs["input_ids"].shape[-1]
    response = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
    return response


def evaluate_token_on_tasks(model, tokenizer, token_decode, tasks, wrong_task_nums,
                            max_new_tokens, use_hf_template=False):
    results = {}
    for task in tasks:
        task_name = task["task"]
        task_num = int(task_name.replace("task", ""))
        if task_num not in wrong_task_nums:
            continue
        template = task["template1"]
        response = generate_task_response(
            model, tokenizer, template, token_decode, max_new_tokens, use_hf_template)
        is_correct = is_task_correct(task_name, token_decode, response)
        results[task_name] = {"is_correct": is_correct, "response": response}
    return results


def find_optimal_parameters(model, tokenizer, glitch_token_id, glitch_token_decode,
                            similar_tokens, wrong_tasks, tasks, max_new_tokens, use_hf_template):
    wrong_task_nums = [int(t.strip()) for t in wrong_tasks.split(",")]
    wrong_task_names = [f"task{num}" for num in wrong_task_nums]

    alpha_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    k_values = list(range(1, min(5, len(similar_tokens) + 1)))

    if not k_values:
        return {"success": False, "alpha": None, "k": None, "results": None}

    original_embedding = model.get_input_embeddings().weight[glitch_token_id].clone()

    best_alpha = None
    best_k = None
    best_success_count = 0
    best_results = None

    for alpha in alpha_values:
        for k in k_values:
            modified_embedding = create_modified_embedding(
                model, glitch_token_id, similar_tokens, alpha, k, tokenizer)
            apply_embedding_modification(model, glitch_token_id, modified_embedding)

            results = evaluate_token_on_tasks(
                model, tokenizer, glitch_token_decode, tasks, wrong_task_nums,
                max_new_tokens, use_hf_template)

            success_count = sum(1 for tn in wrong_task_names
                                if results.get(tn, {}).get("is_correct", False))

            if success_count > best_success_count:
                best_alpha = alpha
                best_k = k
                best_success_count = success_count
                best_results = results

                if success_count == len(wrong_task_names):
                    apply_embedding_modification(model, glitch_token_id, original_embedding)
                    return {"success": True, "alpha": best_alpha, "k": best_k, "results": best_results}

            apply_embedding_modification(model, glitch_token_id, original_embedding)

    success = best_success_count == len(wrong_task_names) and best_alpha is not None
    return {"success": success, "alpha": best_alpha, "k": best_k, "results": best_results}


def apply_fallback_strategy(model, tokenizer, glitch_token_id, glitch_token_decode,
                            similar_tokens, wrong_tasks, tasks, max_new_tokens, use_hf_template):
    """Fallback strategy: fully replace glitch embedding with similar normal tokens (alpha=1.0)."""
    wrong_task_nums = [int(t.strip()) for t in wrong_tasks.split(",")]
    wrong_task_names = [f"task{num}" for num in wrong_task_nums]

    if not similar_tokens:
        return {"success": False, "alpha": None, "k": None, "results": None}

    k = min(5, len(similar_tokens))
    modified_embedding = create_modified_embedding(
        model, glitch_token_id, similar_tokens, 1.0, k, tokenizer)
    apply_embedding_modification(model, glitch_token_id, modified_embedding)

    results = evaluate_token_on_tasks(
        model, tokenizer, glitch_token_decode, tasks, wrong_task_nums,
        max_new_tokens, use_hf_template)

    success_count = sum(1 for tn in wrong_task_names
                        if results.get(tn, {}).get("is_correct", False))

    return {
        "success": success_count == len(wrong_task_names),
        "alpha": 1.0,
        "k": k,
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(description="GlitchEdit: Fix glitch tokens in LLM models")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model")
    parser.add_argument("--glitch_tokens", type=str, required=True, help="Path to glitch tokens JSON")
    parser.add_argument("--tasks", type=str, required=True, help="Path to tasks JSON")
    parser.add_argument("--output_path", type=str, required=True, help="Output directory")
    parser.add_argument("--max_tokens", type=int, default=None,
                        help="Maximum number of glitch tokens to repair (default: all)")
    parser.add_argument("--max_new_tokens", type=int, default=128,
                        help="Max new tokens for generation")
    parser.add_argument("--save_every", type=int, default=10,
                        help="Save report every N tokens (default: 10)")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Temperature (only with do_sample=True)")
    parser.add_argument("--use_hf_chat_template", action="store_true",
                        help="Use tokenizer.apply_chat_template() instead of manual Mistral template")

    args = parser.parse_args()
    os.makedirs(args.output_path, exist_ok=True)

    model, tokenizer = load_model_and_tokenizer(args.model_path)

    with open(args.glitch_tokens, 'r') as f:
        glitch_tokens = json.load(f)
    with open(args.tasks, 'r') as f:
        tasks = json.load(f)

    if args.max_tokens:
        glitch_tokens = glitch_tokens[:args.max_tokens]
        print(f"Limiting to first {args.max_tokens} glitch tokens")

    total_to_process = len(glitch_tokens)
    print(f"Total glitch tokens to repair: {total_to_process}")

    report = {
        "summary": {
            "total_tokens_processed": 0,
            "successful_fixed": 0,
            "partial_fixed": 0,
            "fallback_fixed": 0,
            "failed_fixes": 0,
            "skipped_tokens": 0,
            "error_tokens": 0,
            "fix_rate": 0.0,
        },
        "token_results": []
    }
    report_path = os.path.join(args.output_path, "repair_report.json")

    start_time = time.time()

    for i, glitch_token in enumerate(glitch_tokens):
        token_id_str = glitch_token["tokenID"]
        token_decode = glitch_token["tokenDecode"]
        wrong_tasks = glitch_token["wrong_tasks"]

        print(f"\n[{i+1}/{total_to_process}] Processing token ID={token_id_str}", flush=True)

        try:
            token_id = int(token_id_str)
        except ValueError:
            token_result = {
                "tokenID": token_id_str,
                "original_wrong_tasks": wrong_tasks,
                "repair_status": "error",
                "error_message": "Invalid token ID"
            }
            report["token_results"].append(token_result)
            report["summary"]["error_tokens"] += 1
            report["summary"]["total_tokens_processed"] += 1
            continue

        if not wrong_tasks or wrong_tasks.strip() == "":
            token_result = {
                "tokenID": token_id_str,
                "original_wrong_tasks": wrong_tasks,
                "repair_status": "skipped",
            }
            report["token_results"].append(token_result)
            report["summary"]["skipped_tokens"] += 1
            report["summary"]["total_tokens_processed"] += 1
            continue

        try:
            similar_tokens = find_similar_normal_tokens(token_decode, tokenizer)

            if not similar_tokens:
                token_result = {
                    "tokenID": token_id_str,
                    "original_wrong_tasks": wrong_tasks,
                    "repair_status": "failed",
                }
                report["token_results"].append(token_result)
                report["summary"]["failed_fixes"] += 1
                report["summary"]["total_tokens_processed"] += 1
                continue

            optimal_params = find_optimal_parameters(
                model, tokenizer, token_id, token_decode, similar_tokens,
                wrong_tasks, tasks, args.max_new_tokens, args.use_hf_chat_template)

            wrong_task_nums = [int(t.strip()) for t in wrong_tasks.split(",")]

            if optimal_params["success"]:
                alpha = optimal_params["alpha"]
                k = optimal_params["k"]
                modified_embedding = create_modified_embedding(
                    model, token_id, similar_tokens, alpha, k, tokenizer)
                apply_embedding_modification(model, token_id, modified_embedding)

                token_result = {
                    "tokenID": token_id_str,
                    "original_wrong_tasks": wrong_tasks,
                    "repair_status": "success",
                    "repair_details": {"alpha": alpha, "k": k},
                    "fixed_tasks": [f"task{n}" for n in wrong_task_nums],
                    "remaining_wrong_tasks": [],
                }
                report["summary"]["successful_fixed"] += 1

            elif optimal_params["alpha"] is not None:
                alpha = optimal_params["alpha"]
                k = optimal_params["k"]
                modified_embedding = create_modified_embedding(
                    model, token_id, similar_tokens, alpha, k, tokenizer)
                apply_embedding_modification(model, token_id, modified_embedding)

                fixed_tasks = []
                remaining = []
                for task_num in wrong_task_nums:
                    tn = f"task{task_num}"
                    if optimal_params["results"].get(tn, {}).get("is_correct", False):
                        fixed_tasks.append(tn)
                    else:
                        remaining.append(tn)

                token_result = {
                    "tokenID": token_id_str,
                    "original_wrong_tasks": wrong_tasks,
                    "repair_status": "partial",
                    "repair_details": {"alpha": alpha, "k": k},
                    "fixed_tasks": fixed_tasks,
                    "remaining_wrong_tasks": remaining,
                }
                report["summary"]["partial_fixed"] += 1
            else:
                fallback_result = apply_fallback_strategy(
                    model, tokenizer, token_id, token_decode, similar_tokens,
                    wrong_tasks, tasks, args.max_new_tokens, args.use_hf_chat_template)

                if fallback_result["success"]:
                    token_result = {
                        "tokenID": token_id_str,
                        "original_wrong_tasks": wrong_tasks,
                        "repair_status": "fallback",
                        "repair_details": {"alpha": 1.0, "k": fallback_result["k"]},
                        "fixed_tasks": [f"task{n}" for n in wrong_task_nums],
                        "remaining_wrong_tasks": [],
                    }
                    report["summary"]["fallback_fixed"] += 1
                else:
                    token_result = {
                        "tokenID": token_id_str,
                        "original_wrong_tasks": wrong_tasks,
                        "repair_status": "failed",
                        "repair_details": {"alpha": 1.0, "k": fallback_result["k"]},
                    }
                    report["summary"]["failed_fixes"] += 1

        except Exception as e:
            token_result = {
                "tokenID": token_id_str,
                "original_wrong_tasks": wrong_tasks,
                "repair_status": "error",
                "error_message": str(e)
            }
            report["summary"]["error_tokens"] += 1

        report["token_results"].append(token_result)
        report["summary"]["total_tokens_processed"] += 1

        processed = report["summary"]["total_tokens_processed"]
        successful = report["summary"]["successful_fixed"]
        partial = report["summary"]["partial_fixed"]
        fallback = report["summary"]["fallback_fixed"]
        failed = report["summary"]["failed_fixes"]
        fixed_total = successful + partial + fallback
        fix_rate = (successful + partial + fallback) / processed if processed > 0 else 0

        report["summary"]["fix_rate"] = round(fix_rate * 100, 2)

        elapsed = time.time() - start_time
        avg_time = elapsed / (i + 1)
        eta_min = avg_time * (total_to_process - i - 1) / 60

        print(f"  Status: {token_result.get('repair_status', 'unknown')} | "
              f"Progress: successful={successful}, partial={partial}, failed={failed} | "
              f"Fix rate: {fix_rate*100:.1f}% | "
              f"Remaining: {total_to_process - i - 1} | ETA: {eta_min:.1f}min", flush=True)

        if (i + 1) % args.save_every == 0 or i == total_to_process - 1:
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)

    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    model_output_path = os.path.join(args.output_path, "repaired_model_final")
    os.makedirs(model_output_path, exist_ok=True)
    print(f"\nSaving repaired model to {model_output_path}...", flush=True)
    model.save_pretrained(model_output_path)
    tokenizer.save_pretrained(model_output_path)

    print(f"\n{'='*60}")
    print(f"REPAIR COMPLETE")
    print(f"{'='*60}")
    print(f"Total processed: {report['summary']['total_tokens_processed']}")
    print(f"Successful: {report['summary']['successful_fixed']}")
    print(f"Partial: {report['summary']['partial_fixed']}")
    print(f"Fallback: {report['summary']['fallback_fixed']}")
    print(f"Failed: {report['summary']['failed_fixes']}")
    print(f"Fix rate: {report['summary']['fix_rate']}%")
    print(f"Report saved to: {report_path}")
    print(f"Model saved to: {model_output_path}")


if __name__ == "__main__":
    main()
