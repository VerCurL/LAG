#!/usr/bin/env python
import argparse
import json
import logging
import random
import sys
import time
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
    masks_for_field = masks[:-1] if masks.shape[0] == time_steps + 1 else masks
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


def window_starts(length, chunk_length, chunk_stride, include_tail):
    length = int(length)
    chunk_length = int(chunk_length)
    chunk_stride = int(chunk_stride)
    if length < chunk_length:
        return []

    starts = list(range(0, length - chunk_length + 1, chunk_stride))
    final_start = length - chunk_length
    if include_tail and starts and starts[-1] != final_start:
        starts.append(final_start)
    return starts


def append_episode_windows(accumulator, npz_path, field_calculator, args):
    obs, actions, masks, snapshots, metadata = load_raw_episode(npz_path)
    obs, actions, threat_targets, attack_targets, temporal_targets = build_targets_for_episode(
        obs=obs,
        actions=actions,
        masks=masks,
        snapshots=snapshots,
        field_calculator=field_calculator,
    )

    metadata["pair_key"] = f"{metadata['ego_model_path']}|{metadata['enm_model_path']}"
    metadata["source_file"] = normalize_path(npz_path)

    starts = window_starts(
        length=obs.shape[0],
        chunk_length=args.chunk_length,
        chunk_stride=args.chunk_stride,
        include_tail=args.include_tail_window,
    )

    for start in starts:
        end = start + int(args.chunk_length)
        accumulator["obs"].append(obs[start:end])
        accumulator["actions"].append(actions[start:end])
        accumulator["threat_targets"].append(threat_targets[start:end])
        accumulator["attack_targets"].append(attack_targets[start:end])
        accumulator["temporal_targets"].append(temporal_targets[start:end])
        accumulator["source_files"].append(metadata["source_file"])
        accumulator["episode_ids"].append(metadata["episode_id"])
        accumulator["pair_keys"].append(metadata["pair_key"])
        accumulator["pair_types"].append(metadata["pair_type"])
        accumulator["ego_model_paths"].append(metadata["ego_model_path"])
        accumulator["enm_model_paths"].append(metadata["enm_model_path"])
        accumulator["ego_levels"].append(metadata["ego_level"])
        accumulator["enm_levels"].append(metadata["enm_level"])
        accumulator["scenario_ids"].append(metadata["scenario_id"])
        accumulator["scenario_buckets"].append(metadata["scenario_bucket"])
        accumulator["random_seeds"].append(metadata["random_seed"])
        accumulator["window_starts"].append(int(start))
        accumulator["window_ends"].append(int(end))

    return int(obs.shape[0]), len(starts), metadata


def new_accumulator():
    return {
        "obs": [],
        "actions": [],
        "threat_targets": [],
        "attack_targets": [],
        "temporal_targets": [],
        "source_files": [],
        "episode_ids": [],
        "pair_keys": [],
        "pair_types": [],
        "ego_model_paths": [],
        "enm_model_paths": [],
        "ego_levels": [],
        "enm_levels": [],
        "scenario_ids": [],
        "scenario_buckets": [],
        "random_seeds": [],
        "window_starts": [],
        "window_ends": [],
    }


def take_items(values, indices):
    return [values[index] for index in indices]


def split_indices(num_items, train_ratio, val_ratio, test_ratio, split_seed):
    if num_items <= 0:
        raise ValueError("No windows were generated.")

    ratios = np.asarray([train_ratio, val_ratio, test_ratio], dtype=np.float64)
    if np.any(ratios < 0.0) or float(ratios.sum()) <= 0.0:
        raise ValueError("Split ratios must be non-negative and contain at least one positive value.")
    ratios = ratios / ratios.sum()

    indices = list(range(num_items))
    random.Random(split_seed).shuffle(indices)

    train_count = int(round(num_items * ratios[0]))
    val_count = int(round(num_items * ratios[1]))
    train_count = min(max(train_count, 0), num_items)
    val_count = min(max(val_count, 0), num_items - train_count)

    splits = {
        "train": indices[:train_count],
        "val": indices[train_count:train_count + val_count],
        "test": indices[train_count + val_count:],
    }
    if not splits["train"] or not splits["val"] or not splits["test"]:
        raise ValueError(
            f"Each split must contain at least one window, got "
            f"train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}."
        )
    return splits


def pack_split(accumulator, indices, chunk_length):
    if not indices:
        raise ValueError("Cannot pack an empty split.")

    obs = np.stack(take_items(accumulator["obs"], indices), axis=0).astype(np.float32, copy=False)
    actions = np.stack(take_items(accumulator["actions"], indices), axis=0).astype(np.float32, copy=False)
    threat_targets = np.stack(take_items(accumulator["threat_targets"], indices), axis=0).astype(np.float32, copy=False)
    attack_targets = np.stack(take_items(accumulator["attack_targets"], indices), axis=0).astype(np.float32, copy=False)
    temporal_targets = np.stack(take_items(accumulator["temporal_targets"], indices), axis=0).astype(np.float32, copy=False)

    return {
        "obs": obs,
        "actions": actions,
        "threat_targets": threat_targets,
        "attack_targets": attack_targets,
        "temporal_targets": temporal_targets,
        "window_lengths": np.full((len(indices),), int(chunk_length), dtype=np.int32),
        "source_files": np.asarray(take_items(accumulator["source_files"], indices), dtype=object),
        "episode_ids": np.asarray(take_items(accumulator["episode_ids"], indices), dtype=np.int32),
        "pair_keys": np.asarray(take_items(accumulator["pair_keys"], indices), dtype=object),
        "pair_types": np.asarray(take_items(accumulator["pair_types"], indices), dtype=object),
        "ego_model_paths": np.asarray(take_items(accumulator["ego_model_paths"], indices), dtype=object),
        "enm_model_paths": np.asarray(take_items(accumulator["enm_model_paths"], indices), dtype=object),
        "ego_levels": np.asarray(take_items(accumulator["ego_levels"], indices), dtype=object),
        "enm_levels": np.asarray(take_items(accumulator["enm_levels"], indices), dtype=object),
        "scenario_ids": np.asarray(take_items(accumulator["scenario_ids"], indices), dtype=object),
        "scenario_buckets": np.asarray(take_items(accumulator["scenario_buckets"], indices), dtype=object),
        "random_seeds": np.asarray(take_items(accumulator["random_seeds"], indices), dtype=np.int32),
        "window_starts": np.asarray(take_items(accumulator["window_starts"], indices), dtype=np.int32),
        "window_ends": np.asarray(take_items(accumulator["window_ends"], indices), dtype=np.int32),
    }


def save_dataset(path, dataset):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **dataset)


def split_summary(dataset):
    return {
        "windows": int(dataset["obs"].shape[0]),
        "window_length": int(dataset["obs"].shape[1]),
        "steps": int(dataset["obs"].shape[0] * dataset["obs"].shape[1]),
        "obs_shape": list(dataset["obs"].shape),
        "actions_shape": list(dataset["actions"].shape),
    }


def get_parser():
    parser = argparse.ArgumentParser(description="Build AeroTAF fixed-window targets and shuffled train/val/test splits.")
    parser.add_argument("--raw-dir", type=str, required=True, help="Directory containing raw episode_*.npz files.")
    parser.add_argument("--output-dir", type=str, default="", help="Output directory for processed split datasets.")
    parser.add_argument("--file-pattern", type=str, default="episode_*.npz", help="Raw file pattern.")
    parser.add_argument("--split-seed", type=int, default=1, help="Random seed for window-level shuffle and split.")
    parser.add_argument("--train-ratio", type=float, default=0.80, help="Train window ratio.")
    parser.add_argument("--val-ratio", type=float, default=0.10, help="Validation window ratio.")
    parser.add_argument("--test-ratio", type=float, default=0.10, help="Test window ratio.")
    parser.add_argument("--chunk-length", type=int, default=50, help="Fixed window length saved into the split npz files.")
    parser.add_argument("--chunk-stride", type=int, default=30, help="Stride used to generate fixed windows from each raw episode.")
    parser.add_argument("--include-tail-window", action="store_true", default=True, help="Append the final aligned window when the stride does not land on the episode tail.")
    parser.add_argument("--no-include-tail-window", action="store_false", dest="include_tail_window", help="Disable final aligned tail window.")
    parser.add_argument("--field-k-step", type=int, default=20, help="Future horizon K for AeroTAF target calculation.")
    parser.add_argument("--field-gamma", type=float, default=0.95, help="Discount gamma used in K-step target calculation.")
    parser.add_argument("--ego-team", type=float, default=0.0, help="Ego team id used by FieldCalculator.")
    parser.add_argument("--r-min", type=float, default=4000.0, help="Minimum effective attack range for FieldCalculator.")
    parser.add_argument("--r-attack", type=float, default=14000.0, help="Attack-zone range threshold.")
    parser.add_argument("--r-nez", type=float, default=10000.0, help="No-escape-zone range threshold.")
    parser.add_argument("--theta-attack-deg", type=float, default=60.0, help="Attack-zone angle threshold in degrees.")
    parser.add_argument("--theta-nez-deg", type=float, default=30.0, help="No-escape-zone angle threshold in degrees.")
    return parser


def main(args):
    parser = get_parser()
    all_args = parser.parse_args(args)

    if all_args.chunk_length <= 0:
        raise ValueError("--chunk-length must be positive.")
    if all_args.chunk_stride <= 0:
        raise ValueError("--chunk-stride must be positive.")

    raw_dir = resolve_project_path(all_args.raw_dir)
    if not raw_dir.exists():
        raise FileNotFoundError(f"raw dir not found: {raw_dir}")

    output_dir = (
        resolve_project_path(all_args.output_dir)
        if all_args.output_dir
        else raw_dir.parent / f"processed_windows_K{all_args.field_k_step}_L{all_args.chunk_length}_S{all_args.chunk_stride}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_files = sorted(raw_dir.glob(all_args.file_pattern))
    if not raw_files:
        raise FileNotFoundError(f"No raw npz files found: {raw_dir / all_args.file_pattern}")

    field_calculator = FieldCalculator(
        k_step=all_args.field_k_step,
        gamma=all_args.field_gamma,
        ego_team=all_args.ego_team,
        r_min=all_args.r_min,
        r_attack=all_args.r_attack,
        r_nez=all_args.r_nez,
        theta_attack=np.deg2rad(all_args.theta_attack_deg),
        theta_nez=np.deg2rad(all_args.theta_nez_deg),
    )

    accumulator = new_accumulator()
    failed_files = []
    total_steps = 0

    logging.info(f"Raw dir    : {raw_dir}")
    logging.info(f"Output dir : {output_dir}")
    logging.info(f"Raw files  : {len(raw_files)}")
    logging.info(f"Window     : length={all_args.chunk_length}, stride={all_args.chunk_stride}")

    for index, npz_path in enumerate(raw_files, start=1):
        try:
            steps, windows, metadata = append_episode_windows(accumulator, npz_path, field_calculator, all_args)
            total_steps += steps
            logging.info(
                f"[{index}/{len(raw_files)}] ok: {npz_path.name}, "
                f"T={steps}, windows={windows}, pair_type={metadata['pair_type']}, seed={metadata['random_seed']}"
            )
        except Exception as exc:
            failed_files.append({"source_file": normalize_path(npz_path), "error": repr(exc)})
            logging.info(f"[{index}/{len(raw_files)}] failed: {npz_path.name}, error={repr(exc)}")

    total_windows = len(accumulator["obs"])
    if total_windows <= 0:
        raise RuntimeError("No valid windows were generated.")

    logging.info("")
    logging.info("[stage] shuffle windows and split ...")
    split_index = split_indices(
        num_items=total_windows,
        train_ratio=all_args.train_ratio,
        val_ratio=all_args.val_ratio,
        test_ratio=all_args.test_ratio,
        split_seed=all_args.split_seed,
    )
    split_info = {
        "num_windows_total": int(total_windows),
        "num_windows_train": int(len(split_index["train"])),
        "num_windows_val": int(len(split_index["val"])),
        "num_windows_test": int(len(split_index["test"])),
        "train_ratio_effective": len(split_index["train"]) / total_windows,
        "val_ratio_effective": len(split_index["val"]) / total_windows,
        "test_ratio_effective": len(split_index["test"]) / total_windows,
    }
    logging.info(f"[stage] split done: {split_info}")

    logging.info("[stage] pack split datasets ...")
    datasets = {
        "train": pack_split(accumulator, split_index["train"], all_args.chunk_length),
        "val": pack_split(accumulator, split_index["val"], all_args.chunk_length),
        "test": pack_split(accumulator, split_index["test"], all_args.chunk_length),
    }

    logging.info("[stage] save split datasets ...")
    save_dataset(output_dir / "train.npz", datasets["train"])
    save_dataset(output_dir / "val.npz", datasets["val"])
    save_dataset(output_dir / "test.npz", datasets["test"])

    manifest = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "split_mode": "shuffled_fixed_windows",
        "raw_dir": normalize_path(raw_dir),
        "output_dir": normalize_path(output_dir),
        "field_k_step": all_args.field_k_step,
        "field_gamma": all_args.field_gamma,
        "field_params": {
            "ego_team": all_args.ego_team,
            "r_min": all_args.r_min,
            "r_attack": all_args.r_attack,
            "r_nez": all_args.r_nez,
            "theta_attack_deg": all_args.theta_attack_deg,
            "theta_nez_deg": all_args.theta_nez_deg,
        },
        "split_seed": all_args.split_seed,
        "train_ratio": all_args.train_ratio,
        "val_ratio": all_args.val_ratio,
        "test_ratio": all_args.test_ratio,
        "chunk_length": all_args.chunk_length,
        "chunk_stride": all_args.chunk_stride,
        "include_tail_window": bool(all_args.include_tail_window),
        "num_raw_files": len(raw_files),
        "num_failed_files": len(failed_files),
        "num_raw_steps": int(total_steps),
        "split_info": split_info,
        "split_summary": {name: split_summary(dataset) for name, dataset in datasets.items()},
        "failed_files": failed_files,
        "outputs": {
            "train": "train.npz",
            "val": "val.npz",
            "test": "test.npz",
        },
    }

    with open(output_dir / "split_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    logging.info("")
    logging.info("Window split summary:")
    for split_name, dataset in datasets.items():
        info = split_summary(dataset)
        logging.info(f"{split_name:5s}: windows={info['windows']} steps={info['steps']} obs={info['obs_shape']}")
    logging.info(f"Saved: {output_dir / 'train.npz'}")
    logging.info(f"Saved: {output_dir / 'val.npz'}")
    logging.info(f"Saved: {output_dir / 'test.npz'}")
    logging.info(f"Saved manifest: {output_dir / 'split_manifest.json'}")


if __name__ == "__main__":
    default_args = [
        "--raw-dir", "datasets/aerotaf/4v4_shoot_mappo_pool_stage1/raw",
        "--output-dir", "datasets/aerotaf/4v4_shoot_mappo_pool_stage1/processed_stage1_K20_L50_S30",
        "--split-seed", "1",
        "--train-ratio", "0.80",
        "--val-ratio", "0.10",
        "--test-ratio", "0.10",
        "--chunk-length", "50",
        "--chunk-stride", "30",
        "--field-k-step", "20",
        "--field-gamma", "0.95",
        "--ego-team", "0.0",
        "--r-min", "4000.0",
        "--r-attack", "14000.0",
        "--r-nez", "10000.0",
        "--theta-attack-deg", "60.0",
        "--theta-nez-deg", "30.0",
    ]
    main(sys.argv[1:] if len(sys.argv) > 1 else default_args)
