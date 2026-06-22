#!/usr/bin/env python
import argparse
import json
import multiprocessing as mp
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.AeroTAF.collector.path_utils import (
    canonicalize_task_key,
    normalize_path,
    resolve_project_path,
    to_project_relative_path,
)
from scripts.AeroTAF.collector.rollout_executor import execute_tasks


def get_parser():
    parser = argparse.ArgumentParser(description="Collect AeroTAF raw data with exhaustive model-vs-model sampling.")
    parser.add_argument("--model-root", nargs="+", required=True, help="Actor checkpoint root directories or explicit .pt files.")
    parser.add_argument("--out-dir", default="datasets/aerotaf/4v4_shoot_mappo_pool/full/raw", help="Raw dataset output directory.")
    parser.add_argument("--device", default="cpu", help="Torch device for policy rollout.")
    parser.add_argument("--max-parallel", type=int, default=20, help="Max parallel rollout workers.")
    parser.add_argument("--base-seed", type=int, default=1, help="Base random seed.")
    parser.add_argument("--seeds-per-pair", type=int, default=5, help="Episodes to collect for each directed model pair.")
    parser.add_argument("--deterministic", action="store_true", default=False, help="Use deterministic actor rollout.")
    parser.add_argument("--scenario-name", default="4v4/ShootMissile/HierarchySelfplay")
    parser.add_argument("--policy-type", default="fkr")
    parser.add_argument("--max-episode-steps", type=int, default=1000)
    parser.add_argument("--num-agents-total", type=int, default=8)
    parser.add_argument("--recurrent-hidden-size-actor", type=int, default=128)
    return parser


def parse_checkpoint_step(path_text):
    match = re.search(r"actor_(\d+)\.pt$", Path(path_text).name)
    return int(match.group(1)) if match else -1


def discover_actor_models(model_roots):
    paths = []
    seen = set()
    for root in model_roots:
        path = resolve_project_path(root)
        if path.is_file() and path.suffix == ".pt":
            normalized = normalize_path(path)
            if normalized not in seen:
                seen.add(normalized)
                paths.append(normalized)
            continue

        if path.is_dir():
            for pt_path in sorted(path.glob("actor*.pt")):
                normalized = normalize_path(pt_path)
                if normalized not in seen:
                    seen.add(normalized)
                    paths.append(normalized)

    paths = sorted(paths, key=lambda item: (parse_checkpoint_step(item), item))
    if not paths:
        raise RuntimeError("No actor*.pt models found.")

    return [
        {
            "model_id": f"model_{index:03d}",
            "checkpoint_path": path_text,
            "checkpoint_name": Path(path_text).name,
            "checkpoint_step": parse_checkpoint_step(path_text),
            "source_run": normalize_path(resolve_project_path(path_text).parent),
        }
        for index, path_text in enumerate(paths, start=1)
    ]


def build_scenarios():
    return [
        {
            "scenario_id": "full_default_random_reset",
            "scenario_bucket": "default_random_reset",
            "fix_position": False,
            "description": "Use the environment's built-in random reset.",
        }
    ]


def build_task_key(task):
    return canonicalize_task_key("|".join([
        normalize_path(task["ego_model_path"]),
        normalize_path(task["enm_model_path"]),
        str(task.get("scenario_id", "")),
        str(task.get("seed", "")),
        str(task.get("task_kind", "collect")),
    ]))


def _as_string(value):
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return str(value.item())
        if value.size == 1:
            return str(value.reshape(-1)[0])
    return str(value)


def load_existing_raw_state(raw_dir):
    completed_keys = set()
    existing_episode_ids = []

    for npz_path in sorted(raw_dir.glob("episode_*.npz")):
        match = re.match(r"episode_(\d+)\.npz$", npz_path.name)
        if match:
            existing_episode_ids.append(int(match.group(1)))

        try:
            with np.load(npz_path, allow_pickle=True) as data:
                if "task_key" in data.files:
                    completed_keys.add(canonicalize_task_key(_as_string(data["task_key"])))
        except Exception:
            continue

    return completed_keys, existing_episode_ids


def extend_history(history_path, rows):
    current_rows = []
    if history_path.exists():
        try:
            with open(history_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, list):
                current_rows = loaded
        except Exception:
            current_rows = []

    current_rows.extend(rows)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(current_rows, f, indent=2, ensure_ascii=False)


def build_common_task_fields(all_args, out_dir):
    return {
        "device": all_args.device,
        "out_dir": to_project_relative_path(out_dir),
        "scenario_name": all_args.scenario_name,
        "policy_type": all_args.policy_type,
        "num_agents_total": all_args.num_agents_total,
        "recurrent_hidden_size_actor": all_args.recurrent_hidden_size_actor,
        "max_episode_steps": all_args.max_episode_steps,
        "deterministic": all_args.deterministic,
    }


def iter_model_pairs(model_registry):
    for ego_model in model_registry:
        for enm_model in model_registry:
            if ego_model["checkpoint_path"] == enm_model["checkpoint_path"]:
                continue
            yield ego_model, enm_model


def build_collection_tasks(all_args, out_dir, completed_keys, existing_episode_ids):
    model_registry = discover_actor_models(all_args.model_root)
    if len(model_registry) < 2:
        raise ValueError("Full sampling requires at least 2 actor models.")

    scenarios = build_scenarios()
    common_task_fields = build_common_task_fields(all_args, out_dir)
    seeds_per_pair = int(all_args.seeds_per_pair)
    if seeds_per_pair <= 0:
        raise ValueError("--seeds-per-pair must be positive.")

    next_episode_id = (max(existing_episode_ids) + 1) if existing_episode_ids else 0
    tasks = []
    task_index = 0
    pending_index = 0

    for ego_model, enm_model in iter_model_pairs(model_registry):
        pair = {
            "ego_model_path": ego_model["checkpoint_path"],
            "enm_model_path": enm_model["checkpoint_path"],
            "ego_model_id": ego_model["model_id"],
            "enm_model_id": enm_model["model_id"],
            "ego_level": "all",
            "enm_level": "all",
            "ego_style": "unknown",
            "enm_style": "unknown",
            "ego_stage_hint": "none",
            "enm_stage_hint": "none",
            "pair_type": "full_directed",
        }
        for scenario in scenarios:
            for _ in range(seeds_per_pair):
                seed = int(all_args.base_seed) + task_index
                task = {
                    "task_kind": "collect",
                    "episode_id": next_episode_id + pending_index,
                    "seed": seed,
                    "save_raw": True,
                    **common_task_fields,
                    **pair,
                    **scenario,
                }
                task["scenario_name"] = all_args.scenario_name
                task["task_key"] = build_task_key(task)
                if canonicalize_task_key(task["task_key"]) not in completed_keys:
                    tasks.append(task)
                    pending_index += 1
                task_index += 1

    return model_registry, scenarios, tasks, task_index


def write_run_summary(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def main(args):
    parser = get_parser()
    all_args = parser.parse_args(args)

    out_dir = resolve_project_path(all_args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    completed_keys, existing_episode_ids = load_existing_raw_state(out_dir)
    model_registry, scenarios, pending_tasks, total_task_count = build_collection_tasks(
        all_args=all_args,
        out_dir=out_dir,
        completed_keys=completed_keys,
        existing_episode_ids=existing_episode_ids,
    )

    print("=" * 72)
    print("AeroTAF Full Pair Collection")
    print("=" * 72)
    print(f"models        : {len(model_registry)}")
    print("pair mode     : directed all-pairs")
    print(f"seeds/pair    : {all_args.seeds_per_pair}")
    print(f"scenario      : {all_args.scenario_name}")
    print(f"out dir       : {out_dir}")
    print(f"total tasks   : {total_task_count}")
    print(f"completed     : {total_task_count - len(pending_tasks)}")
    print(f"pending       : {len(pending_tasks)}")
    print("-" * 72)

    summary_path = out_dir.parent / "full_collection_summary.json"
    results_path = out_dir.parent / "full_collect_results.json"
    write_run_summary(
        summary_path,
        {
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model_roots": all_args.model_root,
            "out_dir": to_project_relative_path(out_dir),
            "model_count": len(model_registry),
            "models": model_registry,
            "pair_mode": "directed",
            "seeds_per_pair": all_args.seeds_per_pair,
            "scenario_ids": [item["scenario_id"] for item in scenarios],
            "scenario_name": all_args.scenario_name,
            "total_task_count": total_task_count,
            "completed_task_count": total_task_count - len(pending_tasks),
            "pending_task_count": len(pending_tasks),
        },
    )

    if not pending_tasks:
        print("No pending collection tasks remain.")
        print(f"Summary saved to: {summary_path}")
        return

    collection_results = execute_tasks(pending_tasks, max_parallel=all_args.max_parallel)
    extend_history(results_path, collection_results)

    ok = sum(1 for row in collection_results if row["status"] == "ok")
    failed = len(collection_results) - ok
    print(f"Collection done: ok={ok}, failed={failed}")
    print(f"Raw dataset saved to: {out_dir}")
    print(f"Summary saved to: {summary_path}")
    print(f"Collection history saved to: {results_path}")


if __name__ == "__main__":
    mp.freeze_support()
    default_args = [
        "--model-root", "scripts/results/MultipleCombat/4v4/ShootMissile/HierarchySelfplay/mappo/A128-C512/run-aerotaf-fkr-300vs500",
        "--out-dir", "datasets/aerotaf/4v4_shoot_mappo_pool/fkr-300vs500/raw",
        "--device", "cpu",
        "--max-parallel", "20",
        "--base-seed", "1",
        "--seeds-per-pair", "1000",
        "--scenario-name", "4v4/ShootMissile/HierarchySelfplay",
        "--policy-type", "fkr",
        "--max-episode-steps", "1000",
        "--num-agents-total", "8",
        "--recurrent-hidden-size-actor", "128",
        # "--deterministic",
    ]
    main(sys.argv[1:] if len(sys.argv) > 1 else default_args)
