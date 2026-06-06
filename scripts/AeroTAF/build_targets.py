#!/usr/bin/env python
import argparse
import gc
import json
import logging
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
from scripts.AeroTAF.data.annotation import annotate_splits, fit_annotation_thresholds
from scripts.AeroTAF.data.schema import AnnotationConfig, BUCKET_NAMES, EVENT_NAMES, bucket_summary
from scripts.AeroTAF.data.split import split_stage1_episodes


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


def load_and_process_episode(npz_path, field_calculator, ego_team):
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
    metadata["ego_team"] = float(ego_team)

    return {
        "obs": obs,
        "actions": actions,
        "threat_targets": threat_targets,
        "attack_targets": attack_targets,
        "temporal_targets": temporal_targets,
        "snapshots": snapshots,
        "episode_length": obs.shape[0],
        "meta": metadata,
    }


def combine_episode_items(items):
    if not items:
        raise ValueError("No episode items to combine.")

    obs = np.concatenate([item["obs"] for item in items], axis=0).astype(np.float32, copy=False)
    actions = np.concatenate([item["actions"] for item in items], axis=0).astype(np.float32, copy=False)
    threat_targets = np.concatenate([item["threat_targets"] for item in items], axis=0).astype(np.float32, copy=False)
    attack_targets = np.concatenate([item["attack_targets"] for item in items], axis=0).astype(np.float32, copy=False)
    temporal_targets = np.concatenate([item["temporal_targets"] for item in items], axis=0).astype(np.float32, copy=False)
    sample_bucket = np.concatenate([item["sample_bucket"] for item in items], axis=0).astype(np.int16, copy=False)
    sample_priority = np.concatenate([item["sample_priority"] for item in items], axis=0).astype(np.float32, copy=False)
    sample_weight = np.concatenate([item["sample_weight"] for item in items], axis=0).astype(np.float32, copy=False)
    event_flags = np.concatenate([item["event_flags"] for item in items], axis=0).astype(np.float32, copy=False)
    event_mask = np.concatenate([item["event_mask"] for item in items], axis=0).astype(np.float32, copy=False)
    field_delta_features = np.concatenate([item["field_delta_features"] for item in items], axis=0).astype(np.float32, copy=False)

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
        "sample_bucket": sample_bucket,
        "sample_priority": sample_priority,
        "sample_weight": sample_weight,
        "event_flags": event_flags,
        "event_mask": event_mask,
        "event_names": np.asarray(EVENT_NAMES, dtype=object),
        "bucket_names": np.asarray([BUCKET_NAMES[i] for i in sorted(BUCKET_NAMES)], dtype=object),
        "field_delta_features": field_delta_features,
        "field_delta_feature_names": np.asarray(
            ["delta_threat", "delta_attack", "future_delta_threat", "future_delta_attack", "action_delta"],
            dtype=object,
        ),
        **metadata,
    }


def save_dataset(path, dataset):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **dataset)


def drop_episode_runtime_fields(items):
    for item in items:
        item.pop("snapshots", None)


def build_and_save_split(output_dir, split_name, items):
    split_path = output_dir / f"{split_name}.npz"
    logging.info(f"[split:{split_name}] combining {len(items)} episodes ...")
    dataset = combine_episode_items(items)
    logging.info(
        f"[split:{split_name}] combined: steps={dataset['obs'].shape[0]}, "
        f"obs={dataset['obs'].shape}, actions={dataset['actions'].shape}"
    )
    logging.info(f"[split:{split_name}] saving: {split_path}")
    save_dataset(split_path, dataset)
    logging.info(f"[split:{split_name}] saved")
    del dataset
    gc.collect()


def build_pair_summary(items):
    summary = defaultdict(int)
    for item in items:
        summary[item["meta"]["pair_type"]] += 1
    return dict(sorted(summary.items()))


def build_annotation_summary(items):
    if not items:
        return {"bucket_counts": {}, "event_counts": {}, "sample_count": 0}
    sample_bucket = np.concatenate([item["sample_bucket"] for item in items], axis=0)
    event_flags = np.concatenate([item["event_flags"] for item in items], axis=0)
    priority = np.concatenate([item["sample_priority"].reshape(-1) for item in items], axis=0)
    weight = np.concatenate([item["sample_weight"].reshape(-1) for item in items], axis=0)
    return {
        "bucket_counts": bucket_summary(sample_bucket),
        "event_counts": {name: int(event_flags[:, idx].sum()) for idx, name in enumerate(EVENT_NAMES)},
        "sample_count": int(sample_bucket.shape[0]),
        "priority_mean": float(np.mean(priority)),
        "weight_mean": float(np.mean(weight)),
    }


def build_annotation_config(args):
    return AnnotationConfig(
        field_k_step=args.field_k_step,
        ego_team=args.ego_team,
        high_threat_floor=args.high_threat_floor,
        high_attack_floor=args.high_attack_floor,
        very_high_threat_floor=args.very_high_threat_floor,
        very_high_attack_floor=args.very_high_attack_floor,
        high_field_percentile=args.high_field_percentile,
        very_high_field_percentile=args.very_high_field_percentile,
        delta_floor=args.delta_floor,
        delta_percentile=args.delta_percentile,
        future_delta_floor=args.future_delta_floor,
        future_delta_percentile=args.future_delta_percentile,
        action_change_percentile=args.action_change_percentile,
        event_pre_steps=args.event_pre_steps,
        event_post_steps=args.event_post_steps,
        r_attack=args.r_attack,
        r_nez=args.r_nez,
        theta_attack_deg=args.theta_attack_deg,
        theta_nez_deg=args.theta_nez_deg,
        priority_base=args.priority_base,
        priority_high_threat_bonus=args.priority_high_threat_bonus,
        priority_high_attack_bonus=args.priority_high_attack_bonus,
        priority_high_change_bonus=args.priority_high_change_bonus,
        priority_event_bonus=args.priority_event_bonus,
        priority_action_change_bonus=args.priority_action_change_bonus,
        priority_min=args.priority_min,
        priority_max=args.priority_max,
        weight_from_priority_scale=args.weight_from_priority_scale,
        weight_min=args.weight_min,
        weight_max=args.weight_max,
    )


def get_parser():
    parser = argparse.ArgumentParser(description="Build AeroTAF targets, split datasets, and annotate samples.")
    parser.add_argument("--raw-dir", type=str, required=True, help="Directory containing raw episode_*.npz files.")
    parser.add_argument("--output-dir", type=str, default="", help="Output directory for processed split datasets.")
    parser.add_argument("--file-pattern", type=str, default="episode_*.npz", help="Raw file pattern.")
    parser.add_argument("--split-seed", type=int, default=1, help="Random seed for split.")
    parser.add_argument("--test-pair-ratio", type=float, default=0.15, help="Directed pair ratio held out for Pair-OOD test.")
    parser.add_argument("--val-seed-ratio", type=float, default=0.2, help="Within seen pair, ratio of episodes used for ID validation.")

    parser.add_argument("--field-k-step", type=int, default=20, help="Future horizon K for AeroTAF target calculation.")
    parser.add_argument("--field-gamma", type=float, default=0.95, help="Discount gamma used in K-step window target calculation.")
    parser.add_argument("--ego-team", type=float, default=0.0, help="Ego team id used by FieldCalculator.")
    parser.add_argument("--r-min", type=float, default=4000.0, help="Minimum effective attack range for FieldCalculator.")
    parser.add_argument("--r-attack", type=float, default=14000.0, help="Attack-zone range threshold.")
    parser.add_argument("--r-nez", type=float, default=10000.0, help="No-escape-zone range threshold.")
    parser.add_argument("--theta-attack-deg", type=float, default=60.0, help="Attack-zone angle threshold in degrees.")
    parser.add_argument("--theta-nez-deg", type=float, default=30.0, help="No-escape-zone angle threshold in degrees.")

    parser.add_argument("--high-threat-floor", type=float, default=0.20, help="Fixed lower bound for high-threat threshold.")
    parser.add_argument("--high-attack-floor", type=float, default=0.15, help="Fixed lower bound for high-attack threshold.")
    parser.add_argument("--very-high-threat-floor", type=float, default=0.35, help="Fixed lower bound for very-high-threat threshold.")
    parser.add_argument("--very-high-attack-floor", type=float, default=0.30, help="Fixed lower bound for very-high-attack threshold.")
    parser.add_argument("--high-field-percentile", type=float, default=75.0, help="Train percentile used for high field thresholds.")
    parser.add_argument("--very-high-field-percentile", type=float, default=90.0, help="Train percentile used for very high field thresholds.")
    parser.add_argument("--delta-floor", type=float, default=0.03, help="Fixed lower bound for one-step delta thresholds.")
    parser.add_argument("--delta-percentile", type=float, default=80.0, help="Train percentile used for one-step delta thresholds.")
    parser.add_argument("--future-delta-floor", type=float, default=0.05, help="Fixed lower bound for future delta thresholds.")
    parser.add_argument("--future-delta-percentile", type=float, default=80.0, help="Train percentile used for future delta thresholds.")
    parser.add_argument("--action-change-percentile", type=float, default=80.0, help="Train percentile used for action-change threshold.")
    parser.add_argument("--event-pre-steps", type=int, default=20, help="Steps before an event marked as event-neighborhood.")
    parser.add_argument("--event-post-steps", type=int, default=20, help="Steps after an event marked as event-neighborhood.")

    parser.add_argument("--priority-base", type=float, default=1.0, help="Base sample priority.")
    parser.add_argument("--priority-high-threat-bonus", type=float, default=1.5, help="Priority bonus for high-threat samples.")
    parser.add_argument("--priority-high-attack-bonus", type=float, default=1.5, help="Priority bonus for high-attack samples.")
    parser.add_argument("--priority-high-change-bonus", type=float, default=2.0, help="Priority bonus for high-change samples.")
    parser.add_argument("--priority-event-bonus", type=float, default=3.0, help="Priority bonus for event-neighborhood samples.")
    parser.add_argument("--priority-action-change-bonus", type=float, default=0.5, help="Priority bonus for action-change samples.")
    parser.add_argument("--priority-min", type=float, default=1.0, help="Minimum clipped sample priority.")
    parser.add_argument("--priority-max", type=float, default=8.0, help="Maximum clipped sample priority.")
    parser.add_argument("--weight-from-priority-scale", type=float, default=0.5, help="Loss-weight scale from priority.")
    parser.add_argument("--weight-min", type=float, default=1.0, help="Minimum clipped sample weight.")
    parser.add_argument("--weight-max", type=float, default=4.0, help="Maximum clipped sample weight.")
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
        else raw_dir.parent / f"processed_stage1_K{all_args.field_k_step}_annotated"
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
    annotation_config = build_annotation_config(all_args)

    episode_items = []
    failed_files = []

    logging.info(f"Raw dir: {raw_dir}")
    logging.info(f"Output dir: {output_dir}")
    logging.info(f"Raw files: {len(raw_files)}")

    for index, npz_path in enumerate(raw_files, start=1):
        try:
            item = load_and_process_episode(npz_path, field_calculator, ego_team=all_args.ego_team)
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

    logging.info("")
    logging.info("[stage] split episodes ...")
    split_result = split_stage1_episodes(
        episode_items=episode_items,
        split_seed=all_args.split_seed,
        test_pair_ratio=all_args.test_pair_ratio,
        val_seed_ratio=all_args.val_seed_ratio,
    )
    logging.info(f"[stage] split done: {split_result['split_info']}")

    logging.info("[stage] fit annotation thresholds on train split ...")
    thresholds = fit_annotation_thresholds(split_result["train"], annotation_config)
    logging.info(f"[stage] thresholds ready: {thresholds}")

    logging.info("[stage] annotate train/val/test splits ...")
    annotated_splits = annotate_splits(split_result, thresholds, annotation_config)
    logging.info("[stage] annotation done")

    logging.info("[stage] drop per-episode snapshots before packing splits ...")
    for split_items in annotated_splits.values():
        drop_episode_runtime_fields(split_items)
    gc.collect()

    logging.info("[stage] pack and save split datasets ...")
    build_and_save_split(output_dir, "train", annotated_splits["train"])
    build_and_save_split(output_dir, "val", annotated_splits["val_id"])
    build_and_save_split(output_dir, "test", annotated_splits["test_pair_ood"])

    logging.info("[stage] build annotation summary ...")
    annotation_summary = {
        "train": build_annotation_summary(annotated_splits["train"]),
        "val_id": build_annotation_summary(annotated_splits["val_id"]),
        "test_pair_ood": build_annotation_summary(annotated_splits["test_pair_ood"]),
    }
    logging.info("[stage] build manifest ...")
    split_manifest = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
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
        "test_pair_ratio": all_args.test_pair_ratio,
        "val_seed_ratio": all_args.val_seed_ratio,
        "num_raw_files": len(raw_files),
        "num_processed_files": len(episode_items),
        "num_failed_files": len(failed_files),
        "split_info": split_result["split_info"],
        "test_pair_keys": split_result.get("test_pair_keys", []),
        "pair_type_episode_counts": {
            "train": build_pair_summary(annotated_splits["train"]),
            "val_id": build_pair_summary(annotated_splits["val_id"]),
            "test_pair_ood": build_pair_summary(annotated_splits["test_pair_ood"]),
        },
        "target_annotation": {
            "config": annotation_config.to_dict(),
            "thresholds": thresholds,
            "bucket_names": BUCKET_NAMES,
            "event_names": EVENT_NAMES,
            "summary": annotation_summary,
        },
        "failed_files": failed_files,
        "outputs": {
            "train": "train.npz",
            "val": "val.npz",
            "test": "test.npz",
        },
    }

    logging.info("[stage] save manifest ...")
    with open(output_dir / "stage1_split_manifest.json", "w", encoding="utf-8") as f:
        json.dump(split_manifest, f, indent=2, ensure_ascii=False)

    logging.info("")
    logging.info("Stage-1 split summary:")
    logging.info(split_result["split_info"])
    logging.info("Annotation thresholds:")
    logging.info(thresholds)
    logging.info("Annotation summary:")
    logging.info(annotation_summary)
    logging.info(f"Saved: {output_dir / 'train.npz'}")
    logging.info(f"Saved: {output_dir / 'val.npz'}")
    logging.info(f"Saved: {output_dir / 'test.npz'}")
    logging.info(f"Saved manifest: {output_dir / 'stage1_split_manifest.json'}")


if __name__ == "__main__":
    default_args = [
        "--raw-dir", "datasets/aerotaf/4v4_shoot_mappo_pool_stage1/raw",
        "--output-dir", "datasets/aerotaf/4v4_shoot_mappo_pool_stage1/processed_stage1_K20_annotated",
        "--split-seed", "1",
        "--test-pair-ratio", "0.05",
        "--val-seed-ratio", "0.15",
        "--field-k-step", "20",
        "--field-gamma", "0.95",
        "--ego-team", "0.0",

        "--r-min", "4000.0",                        # 攻击区最小距离
        "--r-attack", "14000.0",                    # 攻击区最大距离
        "--r-nez", "10000.0",                       # 不可逃逸区距离
        "--theta-attack-deg", "60.0",               # 攻击区角度
        "--theta-nez-deg", "30.0",                  # 不可逃逸区角度
        "--high-threat-floor", "0.20",              # 高威胁样本阈值
        "--high-attack-floor", "0.15",              # 高攻击样本阈值
        "--very-high-threat-floor", "0.35",         # 更高威胁样本阈值
        "--very-high-attack-floor", "0.30",         # 更高攻击样本阈值
        "--high-field-percentile", "75.0",          # 训练集上用多少分位数定义“高 threat / 高 attack”
        "--very-high-field-percentile", "90.0",     # 定义“很高”场值的分位数。
        "--delta-floor", "0.03",                    # 场值变化的固定下限
        "--delta-percentile", "80.0",               # 上一步变化量的分位数阈值
        "--future-delta-floor", "0.05",             # 未来若干步场值变化的固定下限
        "--future-delta-percentile", "80.0",        # 未来变化量的分位数阈值
        "--action-change-percentile", "80.0",       # 动作变化强度的分位数阈值

        "--event-pre-steps", "20",
        "--event-post-steps", "20",

        "--priority-base", "1.0",                   # 每个样本的基础优先级
        "--priority-high-threat-bonus", "1.5",      # 高威胁样本加多少优先级
        "--priority-high-attack-bonus", "1.5",      # 高进攻样本加多少优先级
        "--priority-high-change-bonus", "2.0",      # 高变化样本加多少优先级
        "--priority-event-bonus", "3.0",            # 事件邻域样本加多少优先级
        "--priority-action-change-bonus", "0.5",    # 动作变化大样本加多少优先级
        "--priority-min", "1.0",                    # 优先级裁剪范围
        "--priority-max", "8.0",                    # 优先级裁剪范围
        "--weight-from-priority-scale", "0.5",      # 把 priority 映射成训练 loss 权重时的缩放系数
        "--weight-min", "1.0",                      # loss 权重裁剪范围
        "--weight-max", "4.0",                      # loss 权重裁剪范围
        # "--file-pattern", "episode_*.npz",
    ]
    main(sys.argv[1:] if len(sys.argv) > 1 else default_args)
