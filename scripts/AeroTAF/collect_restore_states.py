#!/usr/bin/env python
import argparse
import json
import logging
import multiprocessing as mp
import sys
import time
import traceback
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

logging.basicConfig(level=logging.INFO, format="%(message)s")

try:
    import numpy as np
    import torch
except ModuleNotFoundError as exc:
    logging.info(f"Error: missing dependency: {exc}")
    logging.info("Please activate the same Python environment used by this project, then run this script again.")
    sys.exit(1)

from envs.JSBSim.envs import MultipleCombatEnv
from envs.JSBSim.situation.extractor import SituationExtractor
from scripts.AeroTAF.build_targets_coma import (
    ActorArgs,
    actor_step,
    make_actor,
    np_scalar_to_string,
    object_array_to_strings,
    set_global_seed,
)
from scripts.AeroTAF.collector.path_utils import normalize_path, resolve_project_path


def save_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def save_npz(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)


def read_split_points(dataset_dir, split_names, max_points):
    all_target_path = dataset_dir / "all_target.npz"
    if not all_target_path.exists():
        raise FileNotFoundError(f"all_target not found: {all_target_path}")

    with np.load(all_target_path, allow_pickle=True) as data:
        required = ["source_files", "raw_file_indices", "time_indices"]
        missing = [key for key in required if key not in data.files]
        if missing:
            raise KeyError(f"{all_target_path} missing keys: {missing}")
        all_target = {
            "source_files": object_array_to_strings(data["source_files"]),
            "raw_file_indices": data["raw_file_indices"].astype(np.int64, copy=False).reshape(-1),
            "time_indices": data["time_indices"].astype(np.int64, copy=False).reshape(-1),
            "episode_ids_per_step": (
                data["episode_ids_per_step"].astype(np.int64, copy=False).reshape(-1)
                if "episode_ids_per_step" in data.files
                else np.full(data["time_indices"].shape[0], -1, dtype=np.int64)
            ),
        }

    grouped = {}
    split_counts = {}
    for split_name in split_names:
        split_path = dataset_dir / f"{split_name}.npz"
        if not split_path.exists():
            raise FileNotFoundError(f"split not found: {split_path}")
        with np.load(split_path, allow_pickle=True) as data:
            rows = data["all_target_indices"].astype(np.int64, copy=False).reshape(-1)
        if max_points > 0:
            rows = rows[: int(max_points)]
        split_counts[split_name] = int(rows.shape[0])

        for row in rows:
            raw_file_index = int(all_target["raw_file_indices"][row])
            time_index = int(all_target["time_indices"][row])
            source_file = normalize_path(all_target["source_files"][raw_file_index])
            key = source_file
            item = grouped.setdefault(
                key,
                {
                    "source_file": source_file,
                    "raw_file_index": raw_file_index,
                    "points_by_time": {},
                },
            )
            point_list = item["points_by_time"].setdefault(time_index, [])
            point_list.append(
                {
                    "split": split_name,
                    "all_target_index": int(row),
                    "episode_id": int(all_target["episode_ids_per_step"][row]),
                }
            )

    return list(grouped.values()), split_counts


def load_raw_for_restore(raw_path):
    with np.load(raw_path, allow_pickle=True) as data:
        actions = data["actions"].astype(np.int64, copy=False)
        all_actions = data["all_actions"].astype(np.int64, copy=False) if "all_actions" in data.files else None
        metadata = {
            "episode_id": int(np_scalar_to_string(data["episode_id"])) if "episode_id" in data.files else -1,
            "random_seed": int(np_scalar_to_string(data["random_seed"])) if "random_seed" in data.files else -1,
            "scenario_name": np_scalar_to_string(data["scenario_name"]) if "scenario_name" in data.files else "",
            "policy_type": np_scalar_to_string(data["policy_type"]) if "policy_type" in data.files else "",
            "ego_model_path": normalize_path(np_scalar_to_string(data["ego_model_path"])) if "ego_model_path" in data.files else "",
            "enm_model_path": normalize_path(np_scalar_to_string(data["enm_model_path"])) if "enm_model_path" in data.files else "",
            "scenario_id": np_scalar_to_string(data["scenario_id"]) if "scenario_id" in data.files else "",
            "scenario_bucket": np_scalar_to_string(data["scenario_bucket"]) if "scenario_bucket" in data.files else "",
            "pair_type": np_scalar_to_string(data["pair_type"]) if "pair_type" in data.files else "",
        }
    return actions, all_actions, metadata


def make_env_and_actors(metadata, args):
    seed = int(metadata["random_seed"])
    set_global_seed(seed)
    device = torch.device(args["device"])
    env = MultipleCombatEnv(
        config_name=metadata["scenario_name"],
        policy_type=metadata["policy_type"],
        algorithm="mappo",
        fix_position=bool(args["fix_position"]),
    )
    if env.situation_extractor is None:
        env.situation_extractor = SituationExtractor()
    env.seed(seed)
    ego_actor = make_actor(env, metadata["ego_model_path"], device)
    enm_actor = make_actor(env, metadata["enm_model_path"], device)
    return env, ego_actor, enm_actor


def collect_episode_restore_states(task):
    env = None
    try:
        torch.set_num_threads(1)
        args = task["args"]
        raw_path = resolve_project_path(task["source_file"])
        raw_ego_actions, raw_all_actions, metadata = load_raw_for_restore(raw_path)
        num_ego_agents = int(args["num_agents_total"]) // 2
        max_t = max(int(t) for t in task["points_by_time"].keys())

        env, ego_actor, enm_actor = make_env_and_actors(metadata, args)
        obs, _ = env.reset()
        ego_obs = obs[:num_ego_agents]
        enm_obs = obs[num_ego_agents:]
        ego_rnn_states = np.zeros((num_ego_agents, 1, int(args["recurrent_hidden_size_actor"])), dtype=np.float32)
        enm_rnn_states = np.zeros_like(ego_rnn_states)
        masks = np.ones((num_ego_agents, 1), dtype=np.float32)

        restore_states = []
        time_indices = []
        point_groups = []

        for t in range(max_t + 1):
            if t in task["points_by_time"]:
                all_action_t = raw_all_actions[t].copy() if raw_all_actions is not None and t < raw_all_actions.shape[0] else None
                extra = {
                    "source_file": normalize_path(raw_path),
                    "raw_file_index": int(task["raw_file_index"]),
                    "episode_id": int(metadata["episode_id"]),
                    "time_index": int(t),
                    "point_refs": task["points_by_time"][t],
                    "ego_obs": ego_obs.astype(np.float32, copy=True),
                    "enm_obs": enm_obs.astype(np.float32, copy=True),
                    "ego_rnn_states": ego_rnn_states.astype(np.float32, copy=True),
                    "enm_rnn_states": enm_rnn_states.astype(np.float32, copy=True),
                    "masks": masks.astype(np.float32, copy=True),
                    "factual_ego_actions": raw_ego_actions[t].astype(np.int64, copy=True) if t < raw_ego_actions.shape[0] else None,
                    "factual_all_actions": all_action_t,
                    "metadata": metadata,
                }
                restore_states.append(env.get_restore_state(extra=extra))
                time_indices.append(int(t))
                point_groups.append(task["points_by_time"][t])

            if t == max_t:
                break

            ego_actions, ego_rnn_states = actor_step(
                ego_actor, ego_obs, ego_rnn_states, masks, bool(args["deterministic"])
            )
            enm_actions, enm_rnn_states = actor_step(
                enm_actor, enm_obs, enm_rnn_states, masks, bool(args["deterministic"])
            )
            actions = np.concatenate((ego_actions, enm_actions), axis=0)
            next_obs, _, _, dones, _ = env.step(actions)
            if bool(np.all(dones)) and t + 1 <= max_t:
                raise RuntimeError(f"episode ended at step={t + 1}, before requested max_t={max_t}")
            masks = np.ones((num_ego_agents, 1), dtype=np.float32)
            ego_obs = next_obs[:num_ego_agents]
            enm_obs = next_obs[num_ego_agents:]

        out_dir = resolve_project_path(args["output_dir"])
        out_path = out_dir / "states" / f"restore_episode_{int(metadata['episode_id']):06d}_{raw_path.stem}.npz"
        save_npz(
            out_path,
            {
                "source_file": np.asarray(normalize_path(raw_path), dtype=object),
                "raw_file_index": np.asarray(int(task["raw_file_index"]), dtype=np.int32),
                "episode_id": np.asarray(int(metadata["episode_id"]), dtype=np.int32),
                "time_indices": np.asarray(time_indices, dtype=np.int32),
                "point_groups": np.asarray(point_groups, dtype=object),
                "restore_states": np.asarray(restore_states, dtype=object),
                "metadata": np.asarray(metadata, dtype=object),
            },
        )
        return {
            "status": "ok",
            "source_file": normalize_path(raw_path),
            "output_file": normalize_path(out_path),
            "episode_id": int(metadata["episode_id"]),
            "states": int(len(restore_states)),
        }
    except BaseException as exc:
        return {
            "status": "failed",
            "source_file": task.get("source_file", ""),
            "error": repr(exc),
            "traceback": traceback.format_exc(),
        }
    finally:
        if env is not None:
            env.close()


def iter_results(tasks, num_workers, chunk_size):
    if num_workers <= 1:
        for task in tasks:
            yield collect_episode_restore_states(task)
        return
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=int(num_workers)) as pool:
        yield from pool.imap_unordered(collect_episode_restore_states, tasks, chunksize=max(1, int(chunk_size)))


def get_parser():
    parser = argparse.ArgumentParser(description="Collect pre-action restore states for AeroTAF val/test points.")
    parser.add_argument("--dataset-dir", type=str, required=True, help="Processed detail dataset directory.")
    parser.add_argument("--output-dir", type=str, default="", help="Output directory. Defaults to dataset_dir/restore_states.")
    parser.add_argument("--splits", type=str, default="val test", help="Space-separated split names.")
    parser.add_argument("--num-agents-total", type=int, default=8, help="Total aircraft count; ego count is total/2.")
    parser.add_argument("--deterministic", action="store_true", default=True, help="Use deterministic policy replay.")
    parser.add_argument("--stochastic", action="store_false", dest="deterministic", help="Use stochastic policy replay.")
    parser.add_argument("--fix-position", action="store_true", default=False, help="Pass fix_position=True to MultipleCombatEnv.")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device for actors.")
    parser.add_argument("--recurrent-hidden-size-actor", type=int, default=128, help="Actor recurrent hidden size.")
    parser.add_argument("--max-points", type=int, default=0, help="Optional cap per split for debugging.")
    parser.add_argument("--num-workers", type=int, default=1, help="Parallel episode workers.")
    parser.add_argument("--task-chunk-size", type=int, default=1, help="Multiprocessing chunk size.")
    parser.add_argument("--log-interval", type=int, default=1, help="Progress log interval in raw episodes.")
    return parser


def main(argv):
    args = get_parser().parse_args(argv)
    if args.num_agents_total % 2 != 0:
        raise ValueError("--num-agents-total must be even")

    dataset_dir = resolve_project_path(args.dataset_dir)
    if not dataset_dir.exists():
        raise FileNotFoundError(f"dataset dir not found: {dataset_dir}")
    output_dir = resolve_project_path(args.output_dir) if args.output_dir else dataset_dir / "restore_states"
    output_dir.mkdir(parents=True, exist_ok=True)

    split_names = [name for name in args.splits.split() if name]
    episode_tasks, split_counts = read_split_points(dataset_dir, split_names, int(args.max_points))
    runner_args = {
        "output_dir": normalize_path(output_dir),
        "num_agents_total": int(args.num_agents_total),
        "deterministic": bool(args.deterministic),
        "fix_position": bool(args.fix_position),
        "device": str(args.device),
        "recurrent_hidden_size_actor": int(args.recurrent_hidden_size_actor),
    }
    for task in episode_tasks:
        task["args"] = runner_args

    logging.info("=" * 72)
    logging.info("AeroTAF Restore State Collector")
    logging.info("=" * 72)
    logging.info(f"dataset dir : {normalize_path(dataset_dir)}")
    logging.info(f"output dir  : {normalize_path(output_dir)}")
    logging.info(f"splits      : {split_names}")
    logging.info(f"split points: {split_counts}")
    logging.info(f"raw episodes: {len(episode_tasks)}")
    logging.info(f"workers     : {args.num_workers}")
    logging.info("-" * 72)

    results = []
    for done_i, result in enumerate(iter_results(episode_tasks, int(args.num_workers), int(args.task_chunk_size)), start=1):
        results.append(result)
        if done_i % max(1, int(args.log_interval)) == 0 or done_i == len(episode_tasks):
            ok = sum(1 for row in results if row["status"] == "ok")
            logging.info(f"progress [{done_i}/{len(episode_tasks)}] ok={ok} failed={done_i - ok}")

    manifest = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_dir": normalize_path(dataset_dir),
        "output_dir": normalize_path(output_dir),
        "splits": split_names,
        "split_counts": split_counts,
        "num_raw_episodes": len(episode_tasks),
        "deterministic": bool(args.deterministic),
        "restore_schema": "MultipleCombatEnv.pre_action.v1",
        "outputs": results,
    }
    manifest_path = output_dir / "restore_states_manifest.json"
    save_json(manifest_path, manifest)
    logging.info(f"Saved manifest: {normalize_path(manifest_path)}")
    logging.info("Done.")


if __name__ == "__main__":
    default_args = [
        "--dataset-dir", "datasets/aerotaf/4v4_shoot_mappo_pool/fkr-300vs500/processed_detail_index_k_target_K50",
        "--splits", "val test",
        "--num-agents-total", "8",
        "--device", "cpu",
        "--num-workers", "20",
    ]
    main(sys.argv[1:] if len(sys.argv) > 1 else default_args)
