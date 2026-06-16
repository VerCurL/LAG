#!/usr/bin/env python
import argparse
import gc
import json
import logging
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
from scripts.AeroTAF.data.annotation import annotate_split_items, fit_detail_thresholds
from scripts.AeroTAF.data.schema import (
    CATEGORY_EVENT,
    CATEGORY_HIGH_CHANGE,
    CATEGORY_HIGH_FIELD,
    CATEGORY_NAMES,
    CATEGORY_STABLE,
    CONDITION_NAMES,
    DetailAnnotationConfig,
    EVENT_NAMES,
    FIELD_DELTA_FEATURE_NAMES,
)
from scripts.AeroTAF.data.split import SPLIT_NAMES, split_indices_by_category


POINT_DATA_KEYS = [
    "threat_targets",
    "attack_targets",
    "instant_threat_fields",
    "instant_attack_fields",
    "label_threat_fields",
    "label_attack_fields",
    "field_delta_features",
    "condition_multi_hot",
    "sample_category",
    "event_flags",
]

POINT_INDEX_KEYS = [
    "global_step_indices",
    "raw_file_indices",
    "episode_row_indices",
    "episode_ids_per_step",
    "time_indices",
    "pair_type_indices",
    "scenario_bucket_indices",
    "random_seeds_per_step",
]

POINT_ROW_KEYS = POINT_DATA_KEYS + POINT_INDEX_KEYS


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

        actions = data["actions"].astype(np.float32, copy=False)
        masks = data["masks"].astype(np.float32, copy=False)
        snapshots = as_snapshot_list(data["snapshots"])

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

    return actions, masks, snapshots, metadata


def build_fields_for_episode(snapshots, masks_for_field, field_calculator):
    time_steps = len(snapshots)
    masks_env = masks_for_field.reshape(time_steps, 1, masks_for_field.shape[1], masks_for_field.shape[2])

    instant_threat = np.zeros(time_steps, dtype=np.float32)
    instant_attack = np.zeros(time_steps, dtype=np.float32)
    threat_targets = np.zeros(time_steps, dtype=np.float32)
    attack_targets = np.zeros(time_steps, dtype=np.float32)

    geom_cache = field_calculator._precompute_all_geometry_vectorized(
        snapshots=[snapshots],
        T=time_steps,
        n_envs=1,
    )

    for t in range(time_steps):
        geom_t = (
            geom_cache["AO"][0, t],
            geom_cache["TA"][0, t],
            geom_cache["R"][0, t],
        )
        prev_snapshot = None
        if t > 0 and not np.all(masks_env[t, 0] <= 0.0):
            prev_snapshot = snapshots[t - 1]

        instant_threat[t], instant_attack[t] = field_calculator.instant_team_field(
            snapshots[t],
            geom_t,
            prev_snapshot=prev_snapshot,
        )

    for start, end in field_calculator._episode_segments(time_steps, masks_env, env_i=0):
        threat_targets[start:end] = field_calculator._discounted_k_window(instant_threat[start:end])
        attack_targets[start:end] = field_calculator._discounted_k_window(instant_attack[start:end])

    return (
        instant_threat.reshape(time_steps, 1).astype(np.float32, copy=False),
        instant_attack.reshape(time_steps, 1).astype(np.float32, copy=False),
        threat_targets.reshape(time_steps, 1).astype(np.float32, copy=False),
        attack_targets.reshape(time_steps, 1).astype(np.float32, copy=False),
    )


def load_and_process_episode(npz_path, field_calculator, ego_team):
    actions, masks, snapshots, metadata = load_raw_episode(npz_path)
    time_steps = actions.shape[0]
    masks_for_field = masks[:-1] if masks.shape[0] == time_steps + 1 else masks

    instant_threat, instant_attack, threat_targets, attack_targets = build_fields_for_episode(
        snapshots=snapshots,
        masks_for_field=masks_for_field,
        field_calculator=field_calculator,
    )

    metadata["pair_key"] = f"{metadata['ego_model_path']}|{metadata['enm_model_path']}"
    metadata["source_file"] = normalize_path(npz_path)
    metadata["ego_team"] = float(ego_team)

    return {
        "instant_threat_fields": instant_threat,
        "instant_attack_fields": instant_attack,
        "threat_targets": threat_targets,
        "attack_targets": attack_targets,
        "snapshots": snapshots,
        "episode_length": int(time_steps),
        "meta": metadata,
    }

def take_meta(items, key, dtype=object, default=""):
    return np.asarray([item["meta"].get(key, default) for item in items], dtype=dtype)


def build_value_index(values):
    values = [str(value) for value in values]
    unique_values = sorted(set(values))
    value_to_index = {value: idx for idx, value in enumerate(unique_values)}
    indices = np.asarray([value_to_index[value] for value in values], dtype=np.int32)
    return np.asarray(unique_values, dtype=object), indices


def repeat_episode_values(items, key, dtype=np.int32, default=-1):
    values = []
    for item in items:
        length = int(item["episode_length"])
        values.append(np.full(length, item["meta"].get(key, default), dtype=dtype))
    return np.concatenate(values, axis=0) if values else np.asarray([], dtype=dtype)


def combine_split_items(items):
    if not items:
        raise ValueError("Cannot combine an empty split.")

    episode_lengths = np.asarray([item["episode_length"] for item in items], dtype=np.int32)
    source_files = take_meta(items, "source_file")
    pair_types, pair_type_indices_episode = build_value_index(take_meta(items, "pair_type"))
    scenario_buckets, scenario_bucket_indices_episode = build_value_index(take_meta(items, "scenario_bucket"))

    global_step_indices = []
    raw_file_indices = []
    episode_row_indices = []
    episode_ids_per_step = []
    time_indices = []
    pair_type_indices = []
    scenario_bucket_indices = []
    start = 0
    for episode_row, item in enumerate(items):
        length = int(item["episode_length"])
        end = start + length
        global_step_indices.append(np.arange(start, end, dtype=np.int64))
        raw_file_indices.append(np.full(length, episode_row, dtype=np.int32))
        episode_row_indices.append(np.full(length, episode_row, dtype=np.int32))
        episode_ids_per_step.append(np.full(length, int(item["meta"].get("episode_id", -1)), dtype=np.int32))
        time_indices.append(np.arange(length, dtype=np.int32))
        pair_type_indices.append(np.full(length, pair_type_indices_episode[episode_row], dtype=np.int32))
        scenario_bucket_indices.append(np.full(length, scenario_bucket_indices_episode[episode_row], dtype=np.int32))
        start = end

    dataset = {
        "threat_targets": np.concatenate([item["threat_targets"] for item in items], axis=0).astype(np.float32, copy=False),
        "attack_targets": np.concatenate([item["attack_targets"] for item in items], axis=0).astype(np.float32, copy=False),
        "instant_threat_fields": np.concatenate([item["instant_threat_fields"] for item in items], axis=0).astype(np.float32, copy=False),
        "instant_attack_fields": np.concatenate([item["instant_attack_fields"] for item in items], axis=0).astype(np.float32, copy=False),
        "label_threat_fields": np.concatenate([item["label_threat_fields"] for item in items], axis=0).astype(np.float32, copy=False),
        "label_attack_fields": np.concatenate([item["label_attack_fields"] for item in items], axis=0).astype(np.float32, copy=False),
        "field_delta_features": np.concatenate([item["field_delta_features"] for item in items], axis=0).astype(np.float32, copy=False),
        "condition_multi_hot": np.concatenate([item["condition_multi_hot"] for item in items], axis=0).astype(np.float32, copy=False),
        "sample_category": np.concatenate([item["sample_category"] for item in items], axis=0).astype(np.int16, copy=False),
        "event_flags": np.concatenate([item["event_flags"] for item in items], axis=0).astype(np.float32, copy=False),
        "global_step_indices": np.concatenate(global_step_indices, axis=0),
        "raw_file_indices": np.concatenate(raw_file_indices, axis=0),
        "episode_row_indices": np.concatenate(episode_row_indices, axis=0),
        "episode_ids_per_step": np.concatenate(episode_ids_per_step, axis=0),
        "time_indices": np.concatenate(time_indices, axis=0),
        "pair_type_indices": np.concatenate(pair_type_indices, axis=0),
        "scenario_bucket_indices": np.concatenate(scenario_bucket_indices, axis=0),
        "random_seeds_per_step": repeat_episode_values(items, "random_seed", dtype=np.int32, default=-1),
        "sample_category_names": np.asarray(CATEGORY_NAMES, dtype=object),
        "condition_names": np.asarray(CONDITION_NAMES, dtype=object),
        "event_names": np.asarray(EVENT_NAMES, dtype=object),
        "field_delta_feature_names": np.asarray(FIELD_DELTA_FEATURE_NAMES, dtype=object),
        "source_files": source_files,
        "episode_lengths": episode_lengths,
        "episode_ids": take_meta(items, "episode_id", dtype=np.int32, default=-1),
        "pair_keys": take_meta(items, "pair_key"),
        "pair_types": pair_types,
        "pair_type_per_episode": take_meta(items, "pair_type"),
        "ego_model_paths": take_meta(items, "ego_model_path"),
        "enm_model_paths": take_meta(items, "enm_model_path"),
        "ego_levels": take_meta(items, "ego_level"),
        "enm_levels": take_meta(items, "enm_level"),
        "ego_styles": take_meta(items, "ego_style"),
        "enm_styles": take_meta(items, "enm_style"),
        "scenario_ids": take_meta(items, "scenario_id"),
        "scenario_buckets": scenario_buckets,
        "scenario_bucket_per_episode": take_meta(items, "scenario_bucket"),
        "random_seeds": take_meta(items, "random_seed", dtype=np.int32, default=-1),
        "point_level_split": np.asarray(False),
        "index_only": np.asarray(True),
        "selected_step_count": np.asarray(int(start), dtype=np.int64),
        "parent_step_count": np.asarray(int(start), dtype=np.int64),
    }
    return dataset


def subset_step_dataset(full_dataset, indices):
    indices = np.asarray(indices, dtype=np.int64).reshape(-1)
    subset = {key: full_dataset[key][indices] for key in POINT_ROW_KEYS}
    subset.update(
        {
            "sample_category_names": full_dataset["sample_category_names"],
            "condition_names": full_dataset["condition_names"],
            "event_names": full_dataset["event_names"],
            "field_delta_feature_names": full_dataset["field_delta_feature_names"],
            "source_files": full_dataset["source_files"],
            "episode_lengths": full_dataset["episode_lengths"],
            "episode_ids": full_dataset["episode_ids"],
            "pair_keys": full_dataset["pair_keys"],
            "pair_types": full_dataset["pair_types"],
            "pair_type_per_episode": full_dataset["pair_type_per_episode"],
            "ego_model_paths": full_dataset["ego_model_paths"],
            "enm_model_paths": full_dataset["enm_model_paths"],
            "ego_levels": full_dataset["ego_levels"],
            "enm_levels": full_dataset["enm_levels"],
            "ego_styles": full_dataset["ego_styles"],
            "enm_styles": full_dataset["enm_styles"],
            "scenario_ids": full_dataset["scenario_ids"],
            "scenario_buckets": full_dataset["scenario_buckets"],
            "scenario_bucket_per_episode": full_dataset["scenario_bucket_per_episode"],
            "random_seeds": full_dataset["random_seeds"],
            "point_level_split": np.asarray(True),
            "index_only": np.asarray(True),
            "parent_step_count": np.asarray(int(full_dataset["sample_category"].shape[0]), dtype=np.int64),
            "selected_step_count": np.asarray(int(indices.shape[0]), dtype=np.int64),
        }
    )
    return subset


def pack_category_dataset(full_dataset, category_id):
    mask = full_dataset["sample_category"] == int(category_id)
    category_dataset = subset_step_dataset(full_dataset, np.flatnonzero(mask))
    category_dataset.update(
        {
            "category_name": np.asarray(CATEGORY_NAMES[int(category_id)], dtype=object),
            "category_id": np.asarray(int(category_id), dtype=np.int16),
        }
    )
    return category_dataset


def category_summary(dataset):
    categories = dataset["sample_category"].reshape(-1)
    condition_multi_hot = dataset["condition_multi_hot"]
    event_flags = dataset["event_flags"]
    return {
        "steps": int(categories.shape[0]),
        "category_counts": {
            name: int(np.sum(categories == idx))
            for idx, name in enumerate(CATEGORY_NAMES)
        },
        "condition_counts": {
            name: int(condition_multi_hot[:, idx].sum())
            for idx, name in enumerate(CONDITION_NAMES)
        },
        "event_counts": {
            name: int(event_flags[:, idx].sum())
            for idx, name in enumerate(EVENT_NAMES)
        },
    }

def save_dataset(path, dataset):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **dataset)


def save_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def drop_snapshots(items):
    for item in items:
        item.pop("snapshots", None)


def get_parser():
    parser = argparse.ArgumentParser(description="Build AeroTAF detailed point-category splits.")
    parser.add_argument("--raw-dir", type=str, required=True, help="Directory containing raw episode_*.npz files.")
    parser.add_argument("--output-dir", type=str, default="", help="Output directory.")
    parser.add_argument("--file-pattern", type=str, default="episode_*.npz", help="Raw file pattern.")
    parser.add_argument("--split-seed", type=int, default=1, help="Random seed for category-wise point split.")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Point ratio assigned to train within each category.")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Point ratio assigned to val within each category.")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="Point ratio assigned to test within each category.")

    parser.add_argument("--field-k-step", type=int, default=20, help="Future horizon K for AeroTAF target calculation.")
    parser.add_argument("--field-gamma", type=float, default=0.95, help="Discount gamma for K-step target calculation.")
    parser.add_argument("--ego-team", type=float, default=0.0, help="Ego team id used by FieldCalculator.")
    parser.add_argument("--r-min", type=float, default=4000.0, help="Minimum effective attack range.")
    parser.add_argument("--r-attack", type=float, default=14000.0, help="Attack-zone range threshold.")
    parser.add_argument("--r-nez", type=float, default=10000.0, help="No-escape-zone range threshold.")
    parser.add_argument("--theta-attack-deg", type=float, default=60.0, help="Attack-zone angle threshold in degrees.")
    parser.add_argument("--theta-nez-deg", type=float, default=30.0, help="No-escape-zone angle threshold in degrees.")

    parser.add_argument("--high-threat-floor", type=float, default=0.20, help="Fixed lower bound for high threat.")
    parser.add_argument("--high-attack-floor", type=float, default=0.15, help="Fixed lower bound for high attack.")
    parser.add_argument("--high-field-percentile", type=float, default=75.0, help="Train percentile for high field.")
    parser.add_argument("--delta-floor", type=float, default=0.03, help="Fixed lower bound for one-step field changes.")
    parser.add_argument("--delta-percentile", type=float, default=80.0, help="Train percentile for one-step changes.")
    return parser


def main(args):
    parser = get_parser()
    all_args = parser.parse_args(args)

    raw_dir = resolve_project_path(all_args.raw_dir)
    if not raw_dir.exists():
        raise FileNotFoundError(f"raw dir not found: {raw_dir}")

    output_dir = (
        resolve_project_path(all_args.output_dir)
        if all_args.output_dir
        else raw_dir.parent / f"processed_detail_index_k_target_K{all_args.field_k_step}"
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
    annotation_config = DetailAnnotationConfig(
        ego_team=all_args.ego_team,
        high_threat_floor=all_args.high_threat_floor,
        high_attack_floor=all_args.high_attack_floor,
        high_field_percentile=all_args.high_field_percentile,
        delta_floor=all_args.delta_floor,
        delta_percentile=all_args.delta_percentile,
    )

    episode_items = []
    failed_files = []
    total_steps = 0

    logging.info("=" * 72)
    logging.info("AeroTAF Detail Target Builder")
    logging.info("=" * 72)
    logging.info(f"raw dir      : {normalize_path(raw_dir)}")
    logging.info(f"output dir   : {normalize_path(output_dir)}")
    logging.info(f"raw files    : {len(raw_files)}")
    logging.info("field source : k-step target")
    logging.info("storage mode : point index only")
    logging.info("-" * 72)

    for index, npz_path in enumerate(raw_files, start=1):
        try:
            item = load_and_process_episode(npz_path, field_calculator, ego_team=all_args.ego_team)
            episode_items.append(item)
            total_steps += int(item["episode_length"])
            logging.info(
                f"[{index}/{len(raw_files)}] ok: {npz_path.name}, "
                f"T={item['episode_length']}, pair_type={item['meta']['pair_type']}, seed={item['meta']['random_seed']}"
            )
        except Exception as exc:
            failed_files.append({"source_file": normalize_path(npz_path), "error": repr(exc)})
            logging.info(f"[{index}/{len(raw_files)}] failed: {npz_path.name}, error={repr(exc)}")

    if not episode_items:
        raise RuntimeError("No valid raw episodes were processed.")

    logging.info("")
    logging.info("[stage] fit detail thresholds on all processed episodes ...")
    thresholds = fit_detail_thresholds(episode_items, annotation_config)
    logging.info(f"[stage] thresholds: {thresholds}")

    logging.info("[stage] annotate exact time points for all episodes ...")
    annotated_items = annotate_split_items(episode_items, thresholds, annotation_config)
    drop_snapshots(annotated_items)
    gc.collect()

    logging.info("[stage] pack all annotated points ...")
    all_dataset = combine_split_items(annotated_items)
    all_summary = category_summary(all_dataset)
    logging.info(
        f"[all] steps={all_summary['steps']} categories={all_summary['category_counts']}"
    )

    logging.info("[stage] category-wise random point split ...")
    split_indices, category_split_counts, normalized_split_ratios = split_indices_by_category(
        categories=all_dataset["sample_category"],
        category_names=CATEGORY_NAMES,
        train_ratio=all_args.train_ratio,
        val_ratio=all_args.val_ratio,
        test_ratio=all_args.test_ratio,
        seed=all_args.split_seed,
    )
    logging.info(
        "[stage] split ratios: "
        f"train={normalized_split_ratios['train']:.4f}, "
        f"val={normalized_split_ratios['val']:.4f}, "
        f"test={normalized_split_ratios['test']:.4f}"
    )
    for category_name in CATEGORY_NAMES:
        logging.info(f"  - {category_name:11s}: {category_split_counts[category_name]}")

    logging.info("[stage] pack and save split datasets ...")
    split_summaries = {}
    category_outputs = {}
    split_step_counts = {}
    for split_name in SPLIT_NAMES:
        dataset = subset_step_dataset(all_dataset, split_indices[split_name])
        split_step_counts[split_name] = int(dataset["sample_category"].shape[0])
        full_path = output_dir / f"{split_name}.npz"
        save_dataset(full_path, dataset)

        split_summaries[split_name] = category_summary(dataset)
        category_outputs[split_name] = {"all": f"{split_name}.npz"}
        logging.info(
            f"[split:{split_name}] saved points: {normalize_path(full_path)} "
            f"| points={dataset['sample_category'].shape[0]} | categories={split_summaries[split_name]['category_counts']}"
        )

        for category_id, category_name in enumerate(CATEGORY_NAMES):
            category_dataset = pack_category_dataset(dataset, category_id)
            category_path = output_dir / f"{split_name}_{category_name}.npz"
            save_dataset(category_path, category_dataset)
            category_outputs[split_name][category_name] = f"{split_name}_{category_name}.npz"
            logging.info(
                f"  - {category_name:11s}: {int(category_dataset['selected_step_count'])} -> "
                f"{normalize_path(category_path)}"
            )
    manifest = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "split_mode": "all_episodes_then_category_wise_random_point_index_split",
        "raw_dir": normalize_path(raw_dir),
        "output_dir": normalize_path(output_dir),
        "file_pattern": all_args.file_pattern,
        "num_raw_files": len(raw_files),
        "num_processed_files": len(episode_items),
        "num_failed_files": len(failed_files),
        "num_raw_steps": int(total_steps),
        "field_k_step": all_args.field_k_step,
        "field_gamma": all_args.field_gamma,
        "index_schema": {
            "raw_file": "source_files[raw_file_indices[i]]",
            "time_step": "time_indices[i]",
            "episode_id": "episode_ids_per_step[i]",
            "category": "sample_category_names[sample_category[i]]",
            "raw_slice_hint": "load raw_file, then slice max(0, time_step-history+1):time_step+1",
        },
        "field_params": {
            "ego_team": all_args.ego_team,
            "r_min": all_args.r_min,
            "r_attack": all_args.r_attack,
            "r_nez": all_args.r_nez,
            "theta_attack_deg": all_args.theta_attack_deg,
            "theta_nez_deg": all_args.theta_nez_deg,
        },
        "detail_annotation": {
            "category_names": CATEGORY_NAMES,
            "condition_names": CONDITION_NAMES,
            "event_names": EVENT_NAMES,
            "field_delta_feature_names": FIELD_DELTA_FEATURE_NAMES,
            "label_field_source": "k_step_target",
            "category_assignment_order": "event > high_change > high_field > stable",
            "no_neighborhood_expansion": True,
            "threshold_fit_scope": "all_processed_points",
            "thresholds": thresholds,
            "threshold_config": {
                "high_threat_floor": all_args.high_threat_floor,
                "high_attack_floor": all_args.high_attack_floor,
                "high_field_percentile": all_args.high_field_percentile,
                "delta_floor": all_args.delta_floor,
                "delta_percentile": all_args.delta_percentile,
            },
        },
        "split_seed": all_args.split_seed,
        "requested_split_ratios": {
            "train": all_args.train_ratio,
            "val": all_args.val_ratio,
            "test": all_args.test_ratio,
        },
        "normalized_split_ratios": normalized_split_ratios,
        "split_step_counts": split_step_counts,
        "category_split_counts": category_split_counts,
        "all_summary": all_summary,
        "split_summaries": split_summaries,
        "outputs": category_outputs,
        "failed_files": failed_files,
    }
    manifest_path = output_dir / "detail_split_manifest.json"
    save_json(manifest_path, manifest)

    logging.info("")
    logging.info("Detail split summary:")
    for split_name, summary in split_summaries.items():
        logging.info(f"{split_name:5s}: steps={summary['steps']} categories={summary['category_counts']}")
    logging.info(f"Saved manifest: {normalize_path(manifest_path)}")
    logging.info("Done.")


if __name__ == "__main__":
    default_args = [
        "--raw-dir", "datasets/aerotaf/4v4_shoot_mappo_pool/stage1/raw",
        "--output-dir", "datasets/aerotaf/4v4_shoot_mappo_pool/stage1/processed_detail_index_k_target_K100",
        "--file-pattern", "episode_*.npz",
        "--split-seed", "1",
        "--train-ratio", "0.8",
        "--val-ratio", "0.1",
        "--test-ratio", "0.1",
        "--field-k-step", "100",
        "--field-gamma", "0.96",
        "--ego-team", "0.0",
        "--r-min", "4000.0",
        "--r-attack", "14000.0",
        "--r-nez", "10000.0",
        "--theta-attack-deg", "60.0",
        "--theta-nez-deg", "30.0",
        "--high-threat-floor", "0.20",
        "--high-attack-floor", "0.15",
        "--high-field-percentile", "80.0",
        "--delta-floor", "0.005",
        "--delta-percentile", "75.0",
    ]
    main(sys.argv[1:] if len(sys.argv) > 1 else default_args)




