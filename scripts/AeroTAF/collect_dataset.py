#!/usr/bin/env python
import argparse
import multiprocessing as mp
import sys
import time
from pathlib import Path

from scripts.AeroTAF.collector.path_utils import (
    canonicalize_task_key,
    get_project_root,
    resolve_project_path,
    to_project_relative_path,
)

ROOT_DIR = get_project_root()
sys.path.append(str(ROOT_DIR))

from scripts.AeroTAF.collector.dataset_manifest import write_stage1_artifacts
from scripts.AeroTAF.collector.model_profiling import (
    build_stage1_profiling_pairs,
    summarize_profiling_results,
)
from scripts.AeroTAF.collector.model_registry import (
    assign_provisional_stages,
    discover_actor_models,
)
from scripts.AeroTAF.collector.model_stratify import assign_strength_tiers
from scripts.AeroTAF.collector.opponent_selection import select_stage1_directed_pairs
from scripts.AeroTAF.collector.rollout_executor import execute_tasks
from scripts.AeroTAF.collector.scenario_sampler import build_stage1_scenarios
from scripts.AeroTAF.collector.task_history import (
    extend_history,
    load_existing_raw_state,
    load_json_list,
    save_json,
)
from scripts.AeroTAF.collector.task_planner import (
    build_stage1_collection_tasks,
    build_stage1_profiling_tasks,
)
def get_parser():
    parser = argparse.ArgumentParser(description="Stage-1 AeroTAF dataset collector.")
    parser.add_argument(
        "--action",
        choices=["profile", "group", "plan", "collect", "all"],
        default="all",
        help="Which stage-1 step to run.",
    )
    parser.add_argument("--model-root", nargs="+", required=True, help="Actor checkpoint root directories or explicit .pt files.")
    parser.add_argument("--out-dir", default="datasets/aerotaf/4v4_shoot_mappo_pool_stage1/raw", help="Raw dataset output directory.")
    parser.add_argument("--metadata-dir", default="", help="Optional metadata directory. Defaults to out_dir.parent / collector_stage1.")
    parser.add_argument("--device", default="cpu", help="Torch device for policy rollout.")
    parser.add_argument("--max-parallel", type=int, default=20, help="Max parallel rollout workers for collection.")
    parser.add_argument("--profiling-max-parallel", type=int, default=10, help="Max parallel rollout workers for profiling.")
    parser.add_argument("--base-seed", type=int, default=1, help="Base random seed.")
    parser.add_argument("--profile-seeds-per-pair", type=int, default=2, help="Profiling seeds for each profiling pair.")
    parser.add_argument("--collect-seeds-per-pair", type=int, default=5, help="Collection seeds for each directed pair.")
    parser.add_argument("--opponents-per-tier", type=int, default=2, help="How many opponents to choose from each tier for every ego model.")
    parser.add_argument("--deterministic", action="store_true", default=False, help="Use deterministic actor rollout.")
    parser.add_argument("--scenario-name", default="4v4/ShootMissile/HierarchySelfplay")
    parser.add_argument("--policy-type", default="fkr")
    parser.add_argument("--max-episode-steps", type=int, default=1000)
    parser.add_argument("--num-agents-total", type=int, default=8)
    parser.add_argument("--recurrent-hidden-size-actor", type=int, default=128)
    parser.add_argument("--rebuild-profiling", action="store_true", default=False, help="Ignore cached profiling results and rerun profiling.")
    parser.add_argument("--force-replan", action="store_true", default=False, help="Rebuild collection plan even if it already exists.")
    return parser


def ensure_stage1_model_count(model_registry):
    if len(model_registry) != 30:
        raise ValueError(
            f"Stage-1 collector expects exactly 30 models for strong/mid/low 10-way split, got {len(model_registry)}."
        )


def enrich_registry_from_profiles(model_registry, model_profiles):
    profile_lookup = {item["checkpoint_path"]: item for item in model_profiles}
    enriched = []
    for entry in model_registry:
        profile = profile_lookup.get(entry["checkpoint_path"], {})
        merged = dict(entry)
        merged.update({
            "profiling_episodes": profile.get("profiling_episodes", 0),
            "wins": profile.get("wins", 0),
            "draws": profile.get("draws", 0),
            "losses": profile.get("losses", 0),
            "win_rate": profile.get("win_rate", 0.0),
            "draw_rate": profile.get("draw_rate", 0.0),
            "loss_rate": profile.get("loss_rate", 0.0),
            "avg_reward_margin": profile.get("avg_reward_margin", 0.0),
            "avg_alive_margin": profile.get("avg_alive_margin", 0.0),
            "avg_episode_steps": profile.get("avg_episode_steps", 0.0),
            "avg_ego_speed_mps": profile.get("avg_ego_speed_mps", 0.0),
            "avg_ego_nearest_enemy_distance_m": profile.get("avg_ego_nearest_enemy_distance_m", 0.0),
            "avg_ego_attack_window_reward": profile.get("avg_ego_attack_window_reward", 0.0),
            "avg_ego_missile_avoid_reward": profile.get("avg_ego_missile_avoid_reward", 0.0),
            "strength_score": profile.get("strength_score", 0.0),
            "strength_tier": profile.get("strength_tier", "unknown"),
            "style_label": profile.get("style_label", "unknown"),
        })
        enriched.append(merged)
    return enriched


def get_stage1_paths(metadata_dir):
    return {
        "registry": metadata_dir / "stage1_model_registry.json",
        "profiles_json": metadata_dir / "stage1_model_profiles.json",
        "profiles_csv": metadata_dir / "stage1_model_profiles.csv",
        "profiling_plan": metadata_dir / "stage1_profiling_plan.json",
        "profiling_results": metadata_dir / "stage1_profiling_results.json",
        "grouped_registry": metadata_dir / "stage1_grouped_registry.json",
        "pair_plan": metadata_dir / "stage1_pair_plan.json",
        "collection_plan": metadata_dir / "stage1_collection_plan.json",
        "manifest": metadata_dir / "stage1_manifest.json",
        "collect_results": metadata_dir / "stage1_collect_results.json",
    }


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


def load_grouped_registry_or_raise(paths):
    grouped_registry = load_json_list(paths["grouped_registry"])
    if not grouped_registry:
        raise RuntimeError(
            f"Missing grouped registry: {paths['grouped_registry']}. "
            f"Please run --action group first."
        )
    return grouped_registry


def load_collection_plan_or_raise(paths):
    collection_plan = load_json_list(paths["collection_plan"])
    if not collection_plan:
        raise RuntimeError(
            f"Missing collection plan: {paths['collection_plan']}. "
            f"Please run --action plan first."
        )
    return collection_plan


def build_manifest(
    all_args,
    out_dir,
    metadata_dir,
    grouped_registry,
    profiling_pairs,
    profiling_tasks,
    directed_pairs,
    collection_tasks,
    scenarios,
    existing_episode_ids,
    next_episode_id,
):
    return {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "stage": "stage1",
        "scenario_name": all_args.scenario_name,
        "policy_type": all_args.policy_type,
        "out_dir": to_project_relative_path(out_dir),
        "metadata_dir": to_project_relative_path(metadata_dir),
        "model_count": len(grouped_registry),
        "tier_counts": {
            "high": sum(1 for item in grouped_registry if item["strength_tier"] == "high"),
            "mid": sum(1 for item in grouped_registry if item["strength_tier"] == "mid"),
            "low": sum(1 for item in grouped_registry if item["strength_tier"] == "low"),
        },
        "profiling_pair_count": len(profiling_pairs),
        "profiling_task_count": len(profiling_tasks),
        "directed_pair_count": len(directed_pairs),
        "collect_task_count_total": len(collection_tasks),
        "existing_episode_count": len(existing_episode_ids),
        "next_episode_id": next_episode_id,
        "scenario_ids": [item["scenario_id"] for item in scenarios],
        "action_config": {
            "profile_seeds_per_pair": all_args.profile_seeds_per_pair,
            "collect_seeds_per_pair": all_args.collect_seeds_per_pair,
            "opponents_per_tier": all_args.opponents_per_tier,
        },
    }


def action_profile(all_args, out_dir, metadata_dir, paths):
    model_registry = discover_actor_models(all_args.model_root)
    ensure_stage1_model_count(model_registry)
    model_registry = assign_provisional_stages(model_registry)

    scenarios = build_stage1_scenarios()
    common_task_fields = build_common_task_fields(all_args, out_dir)
    profiling_pairs = build_stage1_profiling_pairs(model_registry)
    profiling_tasks = build_stage1_profiling_tasks(
        profiling_pairs=profiling_pairs,
        scenarios=scenarios,
        seeds_per_pair=all_args.profile_seeds_per_pair,
        seed_start=all_args.base_seed,
        common_task_fields=common_task_fields,
    )

    if paths["profiling_results"].exists() and not all_args.rebuild_profiling:
        profiling_results = load_json_list(paths["profiling_results"])
        print(f"Reuse cached profiling results: {paths['profiling_results']}")
    else:
        print(f"Run stage-1 profiling: tasks={len(profiling_tasks)}")
        profiling_results = execute_tasks(profiling_tasks, max_parallel=all_args.profiling_max_parallel)
        save_json(paths["profiling_results"], profiling_results)

    save_json(paths["registry"], model_registry)
    save_json(paths["profiling_plan"], profiling_tasks)

    ok = sum(1 for row in profiling_results if row["status"] == "ok")
    failed = len(profiling_results) - ok
    print(f"Stage-1 profiling ready: ok={ok}, failed={failed}")
    print(f"Registry saved to: {paths['registry']}")
    print(f"Profiling results saved to: {paths['profiling_results']}")


def action_group(all_args, out_dir, metadata_dir, paths):
    model_registry = load_json_list(paths["registry"])
    if not model_registry:
        model_registry = discover_actor_models(all_args.model_root)
        ensure_stage1_model_count(model_registry)
        model_registry = assign_provisional_stages(model_registry)
        save_json(paths["registry"], model_registry)

    profiling_results = load_json_list(paths["profiling_results"])
    if not profiling_results:
        raise RuntimeError(
            f"Missing profiling results: {paths['profiling_results']}. "
            f"Please run --action profile first."
        )

    model_profiles = summarize_profiling_results(model_registry, profiling_results)
    model_profiles = assign_strength_tiers(model_profiles, tier_sizes=(10, 10, 10))
    grouped_registry = enrich_registry_from_profiles(model_registry, model_profiles)
    directed_pairs = select_stage1_directed_pairs(
        model_entries=grouped_registry,
        opponents_per_tier=all_args.opponents_per_tier,
        prefer_style_diversity=True,
    )

    save_json(paths["profiles_json"], model_profiles)
    save_json(paths["grouped_registry"], grouped_registry)
    save_json(paths["pair_plan"], directed_pairs)

    tier_counts = {
        "high": sum(1 for item in grouped_registry if item["strength_tier"] == "high"),
        "mid": sum(1 for item in grouped_registry if item["strength_tier"] == "mid"),
        "low": sum(1 for item in grouped_registry if item["strength_tier"] == "low"),
    }
    print(f"Stage-1 grouping ready: high={tier_counts['high']}, mid={tier_counts['mid']}, low={tier_counts['low']}")
    print(f"Directed pair plan saved to: {paths['pair_plan']}")


def action_plan(all_args, out_dir, metadata_dir, paths):
    grouped_registry = load_grouped_registry_or_raise(paths)
    directed_pairs = load_json_list(paths["pair_plan"])
    if not directed_pairs:
        raise RuntimeError(
            f"Missing pair plan: {paths['pair_plan']}. "
            f"Please run --action group first."
        )

    scenarios = build_stage1_scenarios()
    completed_collect_keys, existing_episode_ids = load_existing_raw_state(out_dir)
    next_episode_id = (max(existing_episode_ids) + 1) if existing_episode_ids else 0
    common_task_fields = build_common_task_fields(all_args, out_dir)

    if paths["collection_plan"].exists() and not all_args.force_replan:
        print(f"Collection plan already exists: {paths['collection_plan']}")
        print("Use --force-replan if you want to rebuild it after changing scenario settings.")
        return

    collection_tasks = build_stage1_collection_tasks(
        directed_pairs=directed_pairs,
        scenarios=scenarios,
        seeds_per_pair=all_args.collect_seeds_per_pair,
        seed_start=all_args.base_seed + 100000,
        common_task_fields=common_task_fields,
        start_episode_id=next_episode_id,
        completed_keys=completed_collect_keys,
    )

    model_registry = load_json_list(paths["registry"])
    profiling_plan = load_json_list(paths["profiling_plan"])
    model_profiles = load_json_list(paths["profiles_json"])
    profiling_pairs = build_stage1_profiling_pairs(model_registry) if model_registry else []

    manifest = build_manifest(
        all_args=all_args,
        out_dir=out_dir,
        metadata_dir=metadata_dir,
        grouped_registry=grouped_registry,
        profiling_pairs=profiling_pairs,
        profiling_tasks=profiling_plan,
        directed_pairs=directed_pairs,
        collection_tasks=collection_tasks,
        scenarios=scenarios,
        existing_episode_ids=existing_episode_ids,
        next_episode_id=next_episode_id,
    )

    write_stage1_artifacts(
        metadata_dir=metadata_dir,
        model_registry=grouped_registry,
        model_profiles=model_profiles,
        pair_plan=directed_pairs,
        profiling_plan=profiling_plan,
        collection_plan=collection_tasks,
        manifest=manifest,
    )

    print(f"Stage-1 collection plan ready: tasks={len(collection_tasks)}")
    print(f"Collection plan saved to: {paths['collection_plan']}")
    print(f"Manifest saved to: {paths['manifest']}")


def action_collect(all_args, out_dir, metadata_dir, paths):
    collection_plan = load_collection_plan_or_raise(paths)
    completed_collect_keys, _ = load_existing_raw_state(out_dir)
    pending_tasks = [
        task for task in collection_plan
        if canonicalize_task_key(task["task_key"]) not in completed_collect_keys
    ]

    print(f"Stage-1 collection plan loaded: total={len(collection_plan)}, pending={len(pending_tasks)}")
    if not pending_tasks:
        print("No pending collection tasks remain.")
        print(f"Raw dataset stays at: {out_dir}")
        return

    collection_results = execute_tasks(pending_tasks, max_parallel=all_args.max_parallel)
    extend_history(paths["collect_results"], collection_results)

    ok = sum(1 for row in collection_results if row["status"] == "ok")
    failed = len(collection_results) - ok
    print(f"Stage-1 collection done: ok={ok}, failed={failed}")
    print(f"Raw dataset saved to: {out_dir}")
    print(f"Collection history saved to: {paths['collect_results']}")


def main(args):
    parser = get_parser()
    all_args = parser.parse_args(args)

    out_dir = resolve_project_path(all_args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metadata_dir = (
        resolve_project_path(all_args.metadata_dir)
        if all_args.metadata_dir
        else (out_dir.parent / "collector_stage1")
    )
    metadata_dir.mkdir(parents=True, exist_ok=True)
    paths = get_stage1_paths(metadata_dir)

    if all_args.action == "profile":
        action_profile(all_args, out_dir, metadata_dir, paths)
        return

    if all_args.action == "group":
        action_group(all_args, out_dir, metadata_dir, paths)
        return

    if all_args.action == "plan":
        action_plan(all_args, out_dir, metadata_dir, paths)
        return

    if all_args.action == "collect":
        action_collect(all_args, out_dir, metadata_dir, paths)
        return

    action_profile(all_args, out_dir, metadata_dir, paths)
    action_group(all_args, out_dir, metadata_dir, paths)
    action_plan(all_args, out_dir, metadata_dir, paths)
    action_collect(all_args, out_dir, metadata_dir, paths)


if __name__ == "__main__":
    mp.freeze_support()
    default_args = [
        # "--action", "all",
        # "--action", "profile",
        # "--action", "group",
        # "--action", "plan",
        "--action", "collect",
        "--model-root", "../results/MultipleCombat/4v4/ShootMissile/HierarchySelfplay/mappo/A128-C512/run-aerotaf",
        "--out-dir", "datasets/aerotaf/4v4_shoot_mappo_pool_stage1/raw",
        "--device", "cpu",
        "--max-parallel", "20",
        "--profiling-max-parallel", "10",
        "--base-seed", "1",
        "--profile-seeds-per-pair", "2",
        "--collect-seeds-per-pair", "5",
        "--opponents-per-tier", "2",
        "--scenario-name", "4v4/ShootMissile/HierarchySelfplay",
        "--policy-type", "fkr",
        "--max-episode-steps", "1000",
        "--num-agents-total", "8",
        "--recurrent-hidden-size-actor", "128",
        # "--deterministic",
        # "--rebuild-profiling",
        # "--force-replan",
    ]
    main(sys.argv[1:] if len(sys.argv) > 1 else default_args)
