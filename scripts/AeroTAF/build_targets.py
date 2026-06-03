#!/usr/bin/env python
import argparse
import json
import logging
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

logging.basicConfig(level=logging.INFO, format="%(message)s")

try:
    import numpy as np
except ModuleNotFoundError as exc:
    logging.info(f"Error: missing dependency: {exc}")
    logging.info("Please activate the same Python environment used by this project, then run this script again.")
    sys.exit(1)

from envs.JSBSim.situation.field import FieldCalculator
from scripts.AeroTAF.collector.path_utils import canonicalize_task_key, normalize_path, resolve_project_path


class EpisodeSharedBuffer:
    def __init__(self, actions, masks):
        self.actions = actions
        self.masks = masks


def as_snapshot_list(snapshots_array):
    snapshots = []
    for item in snapshots_array:
        if isinstance(item, np.ndarray) and item.ndim == 0 and item.dtype == object:
            snapshots.append(item.item())
        else:
            snapshots.append(item)
    return snapshots


def as_python_scalar(value):
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return value.item()
        if value.size == 1:
            return value.reshape(-1)[0].item()
    return value


def load_raw_episode(npz_path):
    with np.load(npz_path, allow_pickle=True) as data:
        required_keys = ("obs", "actions", "masks", "snapshots")
        missing_keys = [key for key in required_keys if key not in data.files]
        if missing_keys:
            raise KeyError(f"{npz_path} missing keys: {missing_keys}")

        obs = data["obs"].astype(np.float32, copy=False)
        actions = data["actions"].astype(np.float32, copy=False)
        masks = data["masks"].astype(np.float32, copy=False)
        snapshots = as_snapshot_list(data["snapshots"])

        if obs.shape[0] != actions.shape[0]:
            raise ValueError(f"{npz_path}: obs/actions length mismatch")
        if len(snapshots) != actions.shape[0]:
            raise ValueError(f"{npz_path}: snapshots length mismatch: {len(snapshots)} vs {actions.shape[0]}")
        if masks.shape[0] not in (actions.shape[0], actions.shape[0] + 1):
            raise ValueError(f"{npz_path}: unexpected masks length {masks.shape[0]} for T={actions.shape[0]}")

        metadata = {
            "episode_id": int(as_python_scalar(data["episode_id"])) if "episode_id" in data.files else -1,
            "task_key": canonicalize_task_key(as_python_scalar(data["task_key"])) if "task_key" in data.files else "",
            "task_kind": str(as_python_scalar(data["task_kind"])) if "task_kind" in data.files else "",
            "ego_model_path": normalize_path(as_python_scalar(data["ego_model_path"])) if "ego_model_path" in data.files else "",
            "enm_model_path": normalize_path(as_python_scalar(data["enm_model_path"])) if "enm_model_path" in data.files else "",
            "ego_level": str(as_python_scalar(data["ego_level"])) if "ego_level" in data.files else "",
            "enm_level": str(as_python_scalar(data["enm_level"])) if "enm_level" in data.files else "",
            "ego_style": str(as_python_scalar(data["ego_style"])) if "ego_style" in data.files else "",
            "enm_style": str(as_python_scalar(data["enm_style"])) if "enm_style" in data.files else "",
            "pair_type": str(as_python_scalar(data["pair_type"])) if "pair_type" in data.files else "",
            "scenario_id": str(as_python_scalar(data["scenario_id"])) if "scenario_id" in data.files else "",
            "scenario_bucket": str(as_python_scalar(data["scenario_bucket"])) if "scenario_bucket" in data.files else "",
            "random_seed": int(as_python_scalar(data["random_seed"])) if "random_seed" in data.files else -1,
        }

    return obs, actions, masks, snapshots, metadata


def discounted_k_window_features(values, k_step, gamma):
    """
    values: [L, ...]
    return: [L, ...]

    Uses the same K-step discounted-window normalized average style as
    FieldCalculator._discounted_k_window, but for vector features.
    """
    length = int(values.shape[0])
    out = np.zeros_like(values, dtype=np.float32)
    gamma_k = float(gamma) ** int(k_step)

    nums = np.zeros((length + 1,) + values.shape[1:], dtype=np.float32)
    dens = np.zeros(length + 1, dtype=np.float32)

    for t in range(length - 1, -1, -1):
        num = values[t].astype(np.float32, copy=False) + float(gamma) * nums[t + 1]
        den = 1.0 + float(gamma) * dens[t + 1]

        drop_idx = t + int(k_step)
        if drop_idx < length:
            num -= gamma_k * values[drop_idx].astype(np.float32, copy=False)
            den -= gamma_k

        nums[t] = num
        dens[t] = den
        out[t] = num / (den + 1e-6)

    return out


def build_temporal_targets(obs, actions, masks_for_field, field_calculator):
    """
    obs: [T, N, obs_dim]
    actions: [T, N, act_dim]
    masks_for_field: [T, N, 1]

    returns:
        temporal_targets: [T, N, obs_dim + act_dim]
    """
    time_steps = obs.shape[0]
    temporal_source = np.concatenate((obs, actions), axis=-1).astype(np.float32, copy=False)
    temporal_targets = np.zeros_like(temporal_source, dtype=np.float32)

    masks_env = masks_for_field.reshape(time_steps, 1, masks_for_field.shape[1], masks_for_field.shape[2])
    for start, end in field_calculator._episode_segments(time_steps, masks_env, env_i=0):
        temporal_targets[start:end] = discounted_k_window_features(
            temporal_source[start:end],
            k_step=field_calculator.k_step,
            gamma=field_calculator.gamma,
        )

    return temporal_targets


def build_targets_for_episode(obs, actions, masks, snapshots, field_calculator):
    time_steps, num_agents = actions.shape[:2]

    if masks.shape[0] == time_steps + 1:
        masks_for_field = masks[:-1]
    else:
        masks_for_field = masks

    shared_buffer = EpisodeSharedBuffer(
        actions=actions.reshape(time_steps, 1, num_agents, actions.shape[-1]),
        masks=masks_for_field.reshape(time_steps, 1, num_agents, masks_for_field.shape[-1]),
    )

    threat_targets, attack_targets = field_calculator.build_targets(
        snapshots=[snapshots],
        shared_buffer=shared_buffer,
    )
    temporal_targets = build_temporal_targets(
        obs=obs,
        actions=actions,
        masks_for_field=masks_for_field,
        field_calculator=field_calculator,
    )

    return (
        obs.astype(np.float32, copy=False),
        actions.astype(np.float32, copy=False),
        threat_targets.reshape(time_steps, 1).astype(np.float32, copy=False),
        attack_targets.reshape(time_steps, 1).astype(np.float32, copy=False),
        temporal_targets.astype(np.float32, copy=False),
    )


def load_and_process_episode(npz_path, field_calculator):
    obs, actions, masks, snapshots, metadata = load_raw_episode(npz_path)
    obs, actions, threat_targets, attack_targets, temporal_targets = build_targets_for_episode(
        obs=obs,
        actions=actions,
        masks=masks,
        snapshots=snapshots,
        field_calculator=field_calculator,
    )

    pair_key = f"{metadata['ego_model_path']}|{metadata['enm_model_path']}"
    metadata["pair_key"] = pair_key
    metadata["source_file"] = normalize_path(npz_path)

    return {
        "obs": obs,
        "actions": actions,
        "threat_targets": threat_targets,
        "attack_targets": attack_targets,
        "temporal_targets": temporal_targets,
        "episode_length": obs.shape[0],
        "meta": metadata,
    }


def choose_test_pair_keys(pair_groups, test_pair_ratio, split_seed):
    pair_type_to_keys = defaultdict(list)
    for pair_key, episodes in pair_groups.items():
        pair_type = episodes[0]["meta"].get("pair_type", "unknown")
        pair_type_to_keys[pair_type].append(pair_key)

    rng = random.Random(split_seed)
    test_pair_keys = set()

    for pair_type, keys in pair_type_to_keys.items():
        keys = sorted(keys)
        rng.shuffle(keys)
        if len(keys) <= 1:
            continue

        count = int(round(len(keys) * test_pair_ratio))
        count = max(1, count)
        count = min(count, len(keys) - 1)
        test_pair_keys.update(keys[:count])

    return test_pair_keys


def split_stage1_episodes(episode_items, split_seed, test_pair_ratio, val_seed_ratio):
    pair_groups = defaultdict(list)
    for item in episode_items:
        pair_groups[item["meta"]["pair_key"]].append(item)

    for pair_key in pair_groups:
        pair_groups[pair_key] = sorted(
            pair_groups[pair_key],
            key=lambda item: (
                item["meta"].get("random_seed", -1),
                item["meta"].get("episode_id", -1),
                item["meta"].get("source_file", ""),
            ),
        )

    test_pair_keys = choose_test_pair_keys(
        pair_groups=pair_groups,
        test_pair_ratio=test_pair_ratio,
        split_seed=split_seed,
    )

    train_items = []
    val_id_items = []
    test_pair_ood_items = []

    rng = random.Random(split_seed + 999)

    for pair_key, items in pair_groups.items():
        if pair_key in test_pair_keys:
            test_pair_ood_items.extend(items)
            continue

        pair_items = list(items)
        rng.shuffle(pair_items)

        val_count = int(round(len(pair_items) * val_seed_ratio))
        if len(pair_items) >= 2:
            val_count = max(1, val_count)
            val_count = min(val_count, len(pair_items) - 1)
        else:
            val_count = 0

        val_id_items.extend(pair_items[:val_count])
        train_items.extend(pair_items[val_count:])

    split_info = {
        "num_pairs_total": len(pair_groups),
        "num_pairs_train_seen": len(pair_groups) - len(test_pair_keys),
        "num_pairs_test_pair_ood": len(test_pair_keys),
        "num_episodes_train": len(train_items),
        "num_episodes_val_id": len(val_id_items),
        "num_episodes_test_pair_ood": len(test_pair_ood_items),
    }

    return {
        "train": train_items,
        "val_id": val_id_items,
        "test_pair_ood": test_pair_ood_items,
        "split_info": split_info,
    }


def combine_episode_items(items):
    if not items:
        raise ValueError("No episode items to combine.")

    obs = np.concatenate([item["obs"] for item in items], axis=0).astype(np.float32, copy=False)
    actions = np.concatenate([item["actions"] for item in items], axis=0).astype(np.float32, copy=False)
    threat_targets = np.concatenate([item["threat_targets"] for item in items], axis=0).astype(np.float32, copy=False)
    attack_targets = np.concatenate([item["attack_targets"] for item in items], axis=0).astype(np.float32, copy=False)
    temporal_targets = np.concatenate([item["temporal_targets"] for item in items], axis=0).astype(np.float32, copy=False)

    metadata = {
        "episode_lengths": np.asarray([item["episode_length"] for item in items], dtype=np.int32),
        "source_files": np.asarray([item["meta"]["source_file"] for item in items], dtype=object),
        "episode_ids": np.asarray([item["meta"]["episode_id"] for item in items], dtype=np.int32),
        "pair_keys": np.asarray([item["meta"]["pair_key"] for item in items], dtype=object),
        "pair_types": np.asarray([item["meta"]["pair_type"] for item in items], dtype=object),
        "ego_model_paths": np.asarray([item["meta"]["ego_model_path"] for item in items], dtype=object),
        "enm_model_paths": np.asarray([item["meta"]["enm_model_path"] for item in items], dtype=object),
        "ego_levels": np.asarray([item["meta"]["ego_level"] for item in items], dtype=object),
        "enm_levels": np.asarray([item["meta"]["enm_level"] for item in items], dtype=object),
        "scenario_ids": np.asarray([item["meta"]["scenario_id"] for item in items], dtype=object),
        "scenario_buckets": np.asarray([item["meta"]["scenario_bucket"] for item in items], dtype=object),
        "random_seeds": np.asarray([item["meta"]["random_seed"] for item in items], dtype=np.int32),
    }

    return {
        "obs": obs,
        "actions": actions,
        "threat_targets": threat_targets,
        "attack_targets": attack_targets,
        "temporal_targets": temporal_targets,
        **metadata,
    }


def save_dataset(path, dataset):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **dataset)


def build_pair_summary(items):
    summary = defaultdict(int)
    for item in items:
        summary[item["meta"]["pair_type"]] += 1
    return dict(sorted(summary.items()))


def get_parser():
    parser = argparse.ArgumentParser(description="Build stage-1 AeroTAF targets and split datasets.")
    parser.add_argument("--raw-dir", type=str, required=True, help="Directory containing stage-1 raw episode_*.npz files.")
    parser.add_argument("--output-dir", type=str, default="", help="Output directory for processed split datasets.")
    parser.add_argument("--file-pattern", type=str, default="episode_*.npz", help="Raw file pattern.")
    parser.add_argument("--split-seed", type=int, default=1, help="Random seed for stage-1 split.")
    parser.add_argument("--test-pair-ratio", type=float, default=0.15, help="Directed pair ratio held out for Pair-OOD test.")
    parser.add_argument("--val-seed-ratio", type=float, default=0.2, help="Within seen pair, ratio of episodes used for ID validation.")
    parser.add_argument("--field-k-step", type=int, default=20, help="Future horizon K for AeroTAF target calculation.")
    parser.add_argument("--field-gamma", type=float, default=0.95, help="Discount gamma used in K-step window target calculation.")
    parser.add_argument("--ego-team", type=float, default=0.0, help="Ego team id used by FieldCalculator.")
    return parser


def main(args):
    parser = get_parser()
    all_args = parser.parse_args(args)

    raw_dir = resolve_project_path(all_args.raw_dir)
    if not raw_dir.exists():
        raise FileNotFoundError(f"raw dir not found: {raw_dir}")

    if all_args.output_dir:
        output_dir = resolve_project_path(all_args.output_dir)
    else:
        output_dir = raw_dir.parent / f"processed_stage1_K{all_args.field_k_step}"
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_files = sorted(raw_dir.glob(all_args.file_pattern))
    if not raw_files:
        raise FileNotFoundError(f"No raw npz files found: {raw_dir / all_args.file_pattern}")

    field_calculator = FieldCalculator(
        k_step=all_args.field_k_step,
        gamma=all_args.field_gamma,
        ego_team=all_args.ego_team,
    )

    episode_items = []
    failed_files = []

    logging.info(f"Raw dir: {raw_dir}")
    logging.info(f"Output dir: {output_dir}")
    logging.info(f"Raw files: {len(raw_files)}")

    for index, npz_path in enumerate(raw_files, start=1):
        try:
            item = load_and_process_episode(npz_path, field_calculator)
            episode_items.append(item)
            logging.info(
                f"[{index}/{len(raw_files)}] ok: {npz_path.name}, "
                f"T={item['obs'].shape[0]}, pair_type={item['meta']['pair_type']}, seed={item['meta']['random_seed']}"
            )
        except Exception as exc:
            failed_files.append({"source_file": normalize_path(npz_path), "error": repr(exc)})
            logging.info(f"[{index}/{len(raw_files)}] failed: {npz_path.name}, error={repr(exc)}")

    if not episode_items:
        raise RuntimeError("No valid raw episodes were processed.")

    split_result = split_stage1_episodes(
        episode_items=episode_items,
        split_seed=all_args.split_seed,
        test_pair_ratio=all_args.test_pair_ratio,
        val_seed_ratio=all_args.val_seed_ratio,
    )

    datasets = {
        "train": combine_episode_items(split_result["train"]),
        "val_id": combine_episode_items(split_result["val_id"]),
        "test_pair_ood": combine_episode_items(split_result["test_pair_ood"]),
    }

    save_dataset(output_dir / "train.npz", datasets["train"])
    save_dataset(output_dir / "val_id.npz", datasets["val_id"])
    save_dataset(output_dir / "test_pair_ood.npz", datasets["test_pair_ood"])

    save_dataset(output_dir / "val.npz", datasets["val_id"])
    save_dataset(output_dir / "test.npz", datasets["test_pair_ood"])

    split_manifest = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "raw_dir": normalize_path(raw_dir),
        "output_dir": normalize_path(output_dir),
        "field_k_step": all_args.field_k_step,
        "field_gamma": all_args.field_gamma,
        "split_seed": all_args.split_seed,
        "test_pair_ratio": all_args.test_pair_ratio,
        "val_seed_ratio": all_args.val_seed_ratio,
        "num_raw_files": len(raw_files),
        "num_processed_files": len(episode_items),
        "num_failed_files": len(failed_files),
        "split_info": split_result["split_info"],
        "pair_type_episode_counts": {
            "train": build_pair_summary(split_result["train"]),
            "val_id": build_pair_summary(split_result["val_id"]),
            "test_pair_ood": build_pair_summary(split_result["test_pair_ood"]),
        },
        "failed_files": failed_files,
        "outputs": {
            "train": "train.npz",
            "val_id": "val_id.npz",
            "test_pair_ood": "test_pair_ood.npz",
            "val_alias": "val.npz",
            "test_alias": "test.npz",
        },
    }

    with open(output_dir / "stage1_split_manifest.json", "w", encoding="utf-8") as f:
        json.dump(split_manifest, f, indent=2, ensure_ascii=False)

    logging.info("")
    logging.info("Stage-1 split summary:")
    logging.info(split_result["split_info"])
    logging.info(f"Saved: {output_dir / 'train.npz'}")
    logging.info(f"Saved: {output_dir / 'val_id.npz'}")
    logging.info(f"Saved: {output_dir / 'test_pair_ood.npz'}")
    logging.info(f"Saved: {output_dir / 'val.npz'}")
    logging.info(f"Saved: {output_dir / 'test.npz'}")
    logging.info(f"Saved manifest: {output_dir / 'stage1_split_manifest.json'}")


if __name__ == "__main__":
    default_args = [
        "--raw-dir", "datasets/aerotaf/4v4_shoot_mappo_pool_stage1/raw",
        "--output-dir", "datasets/aerotaf/4v4_shoot_mappo_pool_stage1/processed_stage1_K20",
        "--split-seed", "1",
        "--test-pair-ratio", "0.15",
        "--val-seed-ratio", "0.2",
        "--field-k-step", "20",
        "--field-gamma", "0.95",
        "--ego-team", "0.0",
        # "--file-pattern", "episode_*.npz",
    ]
    main(sys.argv[1:] if len(sys.argv) > 1 else default_args)
