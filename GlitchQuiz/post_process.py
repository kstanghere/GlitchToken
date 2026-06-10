#!/usr/bin/env python3

import argparse
import glob
import json
import os


def merge_shard_files(pattern: str) -> list:
    files = sorted(glob.glob(pattern))
    if not files:
        return []
    merged = []
    for path in files:
        with open(path, "r", encoding="utf-8") as f:
            merged.extend(json.load(f))
    return merged


def dedupe_by_token_id(items: list) -> list:
    seen = {}
    for item in items:
        tid = str(item.get("tokenID", ""))
        if tid not in seen:
            seen[tid] = item
    return list(seen.values())


def compute_detection_stats(glitch_tokens: list) -> dict:
    return {
        "total_glitch_tokens": len(glitch_tokens),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Merge assessment outputs and compute stats")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--record_dir", type=str, required=True,
                        help="Directory containing output/ subfolder with GPU shard files")
    parser.add_argument("--output_subdir", type=str, default="output")
    parser.add_argument("--total_tokens", type=int, default=None,
                        help="Total vocab size for glitch_token_ratio")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = os.path.join(args.record_dir, args.output_subdir)
    os.makedirs(args.record_dir, exist_ok=True)

    print(f"Merging shard files from {output_dir}...")

    glitch_shards = merge_shard_files(os.path.join(output_dir, "glitch_tokens_*.json"))
    glitch_tokens = dedupe_by_token_id(glitch_shards)
    glitch_tokens.sort(key=lambda x: int(x["tokenID"]))

    glitch_path = os.path.join(args.record_dir, f"{args.model_name}_glitch_tokens.json")
    stats_path = os.path.join(args.record_dir, f"{args.model_name}_detection_stats.json")

    with open(glitch_path, "w", encoding="utf-8") as f:
        json.dump(glitch_tokens, f, ensure_ascii=False, indent=2)

    stats = compute_detection_stats(glitch_tokens)

    if args.total_tokens:
        stats["total_vocab_tokens"] = args.total_tokens
        stats["glitch_token_ratio"] = round(
            (len(glitch_tokens) / args.total_tokens) * 100, 2)

    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"Model: {args.model_name}")
    print(f"  Glitch tokens: {len(glitch_tokens)} -> {glitch_path}")
    print(f"  Detection stats: {stats_path}")
    print(f"  Stats: {json.dumps(stats, indent=2)}")


if __name__ == "__main__":
    main()
