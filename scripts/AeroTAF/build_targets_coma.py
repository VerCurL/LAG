#!/usr/bin/env python
import argparse
import json
import logging
import multiprocessing as mp
import random
import sys
import time
import traceback
from argparse import Namespace
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

from algorithms.mappo.ppo_actor import PPOActor as MAPPOActor
from envs.JSBSim.envs import MultipleCombatEnv
from envs.JSBSim.situation.extractor import SituationExtractor
from envs.JSBSim.situation.field import FieldCalculator
from scripts.AeroTAF.build_targets_detail import build_fields_for_episode
from scripts.AeroTAF.collector.path_utils import canonicalize_task_key, normalize_path, resolve_project_path
from scripts.AeroTAF.data.schema import CATEGORY_NAMES


CF_ACTION_NAMES = (
    "previous",
    "no_op",
    "invert_maneuver",
    "invert_heading",
    "invert_altitude",
    "invert_velocity",
)
RENDERED_FACTUAL_TACVIEW_PATHS = set()


class ActorArgs:
    def __init__(self):
        self.gain = 0.01
        self.hidden_size_actor = "128 128"
        self.hidden_size_critic = "512 512"
        self.act_hidden_size_actor = "128 128"
        self.act_hidden_size_critic = "512 512"
        self.activation_id = 1
        self.use_feature_normalization = False
        self.use_recurrent_policy = True
        self.recurrent_hidden_size_actor = 128
        self.recurrent_hidden_size_critic = 512
        self.recurrent_hidden_layers = 1
        self.use_prior = True
        self.num_agents = 4


def t2n(x):
    return x.detach().cpu().numpy()


def set_global_seed(seed):
    seed = int(seed)
    seed32 = seed % (2 ** 32)
    random.seed(seed32)
    np.random.seed(seed32)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_state_dict(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def make_actor(env, model_path, device):
    args = ActorArgs()
    actor = MAPPOActor(args, env.observation_space, env.action_space, device=device)
    actor.load_state_dict(load_state_dict(resolve_project_path(model_path), device))
    actor.eval()
    return actor


def object_array_to_strings(values):
    return np.asarray([str(v) for v in np.asarray(values, dtype=object).reshape(-1)], dtype=object)


def np_scalar_to_string(value):
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return str(value.item())
        if value.size == 1:
            return str(value.reshape(-1)[0].item())
    return str(value)


def as_snapshot_list(snapshots_array):
    snapshots = []
    for item in snapshots_array:
        if isinstance(item, np.ndarray) and item.ndim == 0 and item.dtype == object:
            snapshots.append(item.item())
        else:
            snapshots.append(item)
    return snapshots


def read_all_target(data_dir, split_path):
    with np.load(split_path, allow_pickle=True) as split_data:
        if "all_target_indices" not in split_data.files:
            raise KeyError(f"{split_path} missing all_target_indices")
        split_indices = split_data["all_target_indices"].astype(np.int64, copy=False).reshape(-1)
        all_target_file = (
            np_scalar_to_string(split_data["all_target_file"])
            if "all_target_file" in split_data.files
            else "all_target.npz"
        )

    all_target_path = data_dir / all_target_file
    if not all_target_path.exists():
        all_target_path = resolve_project_path(all_target_file)
    if not all_target_path.exists():
        raise FileNotFoundError(f"all_target file not found: {all_target_path}")

    with np.load(all_target_path, allow_pickle=True) as data:
        required = ["source_files", "raw_file_indices", "time_indices", "sample_category"]
        missing = [key for key in required if key not in data.files]
        if missing:
            raise KeyError(f"{all_target_path} missing keys: {missing}")
        all_target = {
            "path": all_target_path,
            "source_files": object_array_to_strings(data["source_files"]),
            "raw_file_indices": data["raw_file_indices"].astype(np.int64, copy=False).reshape(-1),
            "time_indices": data["time_indices"].astype(np.int64, copy=False).reshape(-1),
            "sample_category": data["sample_category"].astype(np.int64, copy=False).reshape(-1),
        }
        if "episode_ids_per_step" in data.files:
            all_target["episode_ids_per_step"] = data["episode_ids_per_step"].astype(np.int64, copy=False).reshape(-1)
        else:
            all_target["episode_ids_per_step"] = np.full(all_target["time_indices"].shape[0], -1, dtype=np.int64)

    return split_indices, all_target


def load_raw_metadata(raw_path):
    with np.load(raw_path, allow_pickle=True) as data:
        required = ["actions"]
        missing = [key for key in required if key not in data.files]
        if missing:
            raise KeyError(f"{raw_path} missing keys: {missing}")
        actions = data["actions"].astype(np.int64, copy=False)
        all_actions = data["all_actions"].astype(np.int64, copy=False) if "all_actions" in data.files else None
        metadata = {
            "episode_id": int(np_scalar_to_string(data["episode_id"])) if "episode_id" in data.files else -1,
            "task_key": canonicalize_task_key(np_scalar_to_string(data["task_key"])) if "task_key" in data.files else "",
            "task_kind": np_scalar_to_string(data["task_kind"]) if "task_kind" in data.files else "",
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


def load_restore_lookup(restore_dir):
    restore_dir = resolve_project_path(restore_dir)
    state_dir = restore_dir / "states"
    if not state_dir.exists():
        state_dir = restore_dir
    restore_files = sorted(state_dir.glob("restore_episode_*.npz"))
    if not restore_files:
        raise FileNotFoundError(f"No restore_episode_*.npz files found in {state_dir}")

    lookup = {}
    for restore_file in restore_files:
        with np.load(restore_file, allow_pickle=True) as data:
            time_indices = data["time_indices"].astype(np.int64, copy=False).reshape(-1)
            point_groups = list(data["point_groups"])
        for state_index, (time_index, point_group) in enumerate(zip(time_indices, point_groups)):
            if isinstance(point_group, np.ndarray) and point_group.shape == () and point_group.dtype == object:
                point_group = point_group.item()
            for point_ref in point_group:
                if isinstance(point_ref, np.ndarray) and point_ref.shape == () and point_ref.dtype == object:
                    point_ref = point_ref.item()
                lookup[int(point_ref["all_target_index"])] = {
                    "restore_file": normalize_path(restore_file),
                    "state_index": int(state_index),
                    "time_index": int(time_index),
                }
    return lookup


def load_restore_state(restore_file, state_index):
    with np.load(resolve_project_path(restore_file), allow_pickle=True) as data:
        state = data["restore_states"][int(state_index)]
        if isinstance(state, np.ndarray) and state.shape == () and state.dtype == object:
            return state.item()
        return state


def counterfactual_action(kind, current_action, previous_action):
    action = np.asarray(current_action, dtype=np.int64).copy()
    shoot = int(action[3]) if action.shape[0] > 3 else 0

    if kind == "previous":
        action[:3] = np.asarray(previous_action, dtype=np.int64)[:3]
    elif kind == "no_op":
        action[:3] = np.asarray([1, 2, 1], dtype=np.int64)
    elif kind == "invert_maneuver":
        action[0] = 2 - action[0]
        action[1] = 4 - action[1]
        action[2] = 2 - action[2]
    elif kind == "invert_heading":
        action[1] = 4 - action[1]
    elif kind == "invert_altitude":
        action[0] = 2 - action[0]
    elif kind == "invert_velocity":
        action[2] = 2 - action[2]
    else:
        raise ValueError(f"Unknown counterfactual action kind: {kind}")

    if action.shape[0] > 3:
        action[3] = shoot
    return action


def make_env_and_actors(metadata, device, fix_position):
    seed = int(metadata["random_seed"])
    set_global_seed(seed)
    env = MultipleCombatEnv(
        config_name=metadata["scenario_name"],
        policy_type=metadata["policy_type"],
        algorithm="mappo",
        fix_position=fix_position,
    )
    if env.situation_extractor is None:
        env.situation_extractor = SituationExtractor()
    env.seed(seed)
    ego_actor = make_actor(env, metadata["ego_model_path"], device)
    enm_actor = make_actor(env, metadata["enm_model_path"], device)
    return env, ego_actor, enm_actor


def actor_step(actor, obs, rnn_states, masks, deterministic):
    with torch.no_grad():
        actions, _, next_rnn_states = actor(obs, rnn_states, masks, deterministic=deterministic)
    return t2n(actions).astype(np.int64, copy=False), t2n(next_rnn_states)


def get_worker_slot(num_workers):
    if int(num_workers) <= 1:
        return 1
    identity = mp.current_process()._identity
    if not identity:
        return 1
    return ((int(identity[0]) - 1) % int(num_workers)) + 1


def render_rollout_acmi(
    env,
    ego_actor,
    enm_actor,
    ego_obs,
    enm_obs,
    ego_rnn_states,
    enm_rnn_states,
    masks,
    first_step_actions,
    args,
    acmi_path,
):
    acmi_path = resolve_project_path(acmi_path)
    acmi_path.parent.mkdir(parents=True, exist_ok=True)
    env.refresh_records()
    env.render(mode="txt", filepath=str(acmi_path))

    num_ego_agents = int(args.num_agents_total) // 2
    ego_obs = np.asarray(ego_obs, dtype=np.float32).copy()
    enm_obs = np.asarray(enm_obs, dtype=np.float32).copy()
    ego_rnn_states = np.asarray(ego_rnn_states, dtype=np.float32).copy()
    enm_rnn_states = np.asarray(enm_rnn_states, dtype=np.float32).copy()
    masks = np.asarray(masks, dtype=np.float32).copy()

    for step_i in range(int(args.field_k_step)):
        if step_i == 0:
            _, ego_rnn_states = actor_step(ego_actor, ego_obs, ego_rnn_states, masks, bool(args.deterministic))
            _, enm_rnn_states = actor_step(enm_actor, enm_obs, enm_rnn_states, masks, bool(args.deterministic))
            step_actions = np.asarray(first_step_actions, dtype=np.int64)
        else:
            ego_actions, ego_rnn_states = actor_step(ego_actor, ego_obs, ego_rnn_states, masks, bool(args.deterministic))
            enm_actions, enm_rnn_states = actor_step(enm_actor, enm_obs, enm_rnn_states, masks, bool(args.deterministic))
            step_actions = np.concatenate((ego_actions, enm_actions), axis=0)

        next_obs, _, _, dones, _ = env.step(step_actions)
        env.render(mode="txt", filepath=str(acmi_path))
        if bool(np.all(dones)):
            break

        masks = np.ones((num_ego_agents, 1), dtype=np.float32)
        ego_obs = next_obs[:num_ego_agents]
        enm_obs = next_obs[num_ego_agents:]


def replay_to_time(env, ego_actor, enm_actor, target_t, num_ego_agents, recurrent_hidden_size, deterministic):
    obs, _ = env.reset()
    ego_obs = obs[:num_ego_agents]
    enm_obs = obs[num_ego_agents:]
    ego_rnn_states = np.zeros((num_ego_agents, 1, recurrent_hidden_size), dtype=np.float32)
    enm_rnn_states = np.zeros_like(ego_rnn_states)
    masks = np.ones((num_ego_agents, 1), dtype=np.float32)

    for _ in range(int(target_t)):
        ego_actions, ego_rnn_states = actor_step(ego_actor, ego_obs, ego_rnn_states, masks, deterministic)
        enm_actions, enm_rnn_states = actor_step(enm_actor, enm_obs, enm_rnn_states, masks, deterministic)
        actions = np.concatenate((ego_actions, enm_actions), axis=0)
        next_obs, _, _, dones, _ = env.step(actions)
        if bool(np.all(dones)):
            raise RuntimeError(f"episode ended before target_t={target_t}")
        masks = np.ones((num_ego_agents, 1), dtype=np.float32)
        ego_obs = next_obs[:num_ego_agents]
        enm_obs = next_obs[num_ego_agents:]

    return ego_obs, enm_obs, ego_rnn_states, enm_rnn_states, masks


def rollout_counterfactual_label(
    metadata,
    target_t,
    agent_index,
    counterfactual_actions,
    field_calculator,
    args,
    device,
    restore_ref=None,
    factual_actions=None,
    factual_tacview_path="",
    counterfactual_tacview_path="",
):
    env = None
    try:
        num_ego_agents = int(args.num_agents_total) // 2
        restore_state = None
        if restore_ref is not None:
            restore_state = load_restore_state(restore_ref["restore_file"], restore_ref["state_index"])
            metadata = restore_state.get("extra", {}).get("metadata", metadata)
            env, ego_actor, enm_actor = make_env_and_actors(metadata, device, args.fix_position)
            env.set_restore_state(restore_state)
            extra = restore_state.get("extra", {})
            ego_obs = np.asarray(extra["ego_obs"], dtype=np.float32)
            enm_obs = np.asarray(extra["enm_obs"], dtype=np.float32)
            ego_rnn_states = np.asarray(extra["ego_rnn_states"], dtype=np.float32)
            enm_rnn_states = np.asarray(extra["enm_rnn_states"], dtype=np.float32)
            masks = np.asarray(extra["masks"], dtype=np.float32)
        else:
            env, ego_actor, enm_actor = make_env_and_actors(metadata, device, args.fix_position)
            ego_obs, enm_obs, ego_rnn_states, enm_rnn_states, masks = replay_to_time(
                env,
                ego_actor,
                enm_actor,
                target_t,
                num_ego_agents,
                int(args.recurrent_hidden_size_actor),
                bool(args.deterministic),
            )
            restore_state = env.get_restore_state(
                extra={
                    "ego_obs": ego_obs.astype(np.float32, copy=True),
                    "enm_obs": enm_obs.astype(np.float32, copy=True),
                    "ego_rnn_states": ego_rnn_states.astype(np.float32, copy=True),
                    "enm_rnn_states": enm_rnn_states.astype(np.float32, copy=True),
                    "masks": masks.astype(np.float32, copy=True),
                }
            )

        if factual_tacview_path and factual_actions is not None:
            factual_tacview_key = normalize_path(resolve_project_path(factual_tacview_path))
            if factual_tacview_key not in RENDERED_FACTUAL_TACVIEW_PATHS:
                render_rollout_acmi(
                    env,
                    ego_actor,
                    enm_actor,
                    ego_obs,
                    enm_obs,
                    ego_rnn_states,
                    enm_rnn_states,
                    masks,
                    factual_actions,
                    args,
                    factual_tacview_path,
                )
                RENDERED_FACTUAL_TACVIEW_PATHS.add(factual_tacview_key)
            env.set_restore_state(restore_state)
            extra = restore_state.get("extra", {})
            ego_obs = np.asarray(extra["ego_obs"], dtype=np.float32)
            enm_obs = np.asarray(extra["enm_obs"], dtype=np.float32)
            ego_rnn_states = np.asarray(extra["ego_rnn_states"], dtype=np.float32)
            enm_rnn_states = np.asarray(extra["enm_rnn_states"], dtype=np.float32)
            masks = np.asarray(extra["masks"], dtype=np.float32)

        if counterfactual_tacview_path:
            acmi_path = resolve_project_path(counterfactual_tacview_path)
            acmi_path.parent.mkdir(parents=True, exist_ok=True)
            env.refresh_records()
            env.render(mode="txt", filepath=str(acmi_path))

        snapshots = []
        masks_for_field = []
        for step_i in range(int(args.field_k_step)):
            if step_i == 0:
                _, ego_rnn_states = actor_step(ego_actor, ego_obs, ego_rnn_states, masks, bool(args.deterministic))
                _, enm_rnn_states = actor_step(enm_actor, enm_obs, enm_rnn_states, masks, bool(args.deterministic))
                step_actions = np.asarray(counterfactual_actions, dtype=np.int64)
            else:
                ego_actions, ego_rnn_states = actor_step(ego_actor, ego_obs, ego_rnn_states, masks, bool(args.deterministic))
                enm_actions, enm_rnn_states = actor_step(enm_actor, enm_obs, enm_rnn_states, masks, bool(args.deterministic))
                step_actions = np.concatenate((ego_actions, enm_actions), axis=0)

            next_obs, _, _, dones, info = env.step(step_actions)
            snapshot = info.get("AeroTAF_snapshot")
            if snapshot is None:
                raise RuntimeError("AeroTAF_snapshot is missing during counterfactual rollout.")
            snapshots.append(snapshot)
            masks_for_field.append(np.ones((num_ego_agents, 1), dtype=np.float32))
            if counterfactual_tacview_path:
                env.render(mode="txt", filepath=str(resolve_project_path(counterfactual_tacview_path)))

            if bool(np.all(dones)):
                break
            masks = np.ones((num_ego_agents, 1), dtype=np.float32)
            ego_obs = next_obs[:num_ego_agents]
            enm_obs = next_obs[num_ego_agents:]

        if not snapshots:
            raise RuntimeError("counterfactual rollout produced no snapshots")

        masks_for_field = np.asarray(masks_for_field, dtype=np.float32)
        _, _, threat_targets, attack_targets = build_fields_for_episode(
            snapshots=snapshots,
            masks_for_field=masks_for_field,
            field_calculator=field_calculator,
        )
        return float(threat_targets[0, 0]), float(attack_targets[0, 0]), len(snapshots)
    finally:
        if env is not None:
            env.close()


def make_field_calculator(args):
    return FieldCalculator(
        k_step=args.field_k_step,
        gamma=args.field_gamma,
        ego_team=args.ego_team,
        r_min=args.r_min,
        r_attack=args.r_attack,
        r_nez=args.r_nez,
        theta_attack=np.deg2rad(args.theta_attack_deg),
        theta_nez=np.deg2rad(args.theta_nez_deg),
    )


def run_counterfactual_task(task):
    try:
        torch.set_num_threads(1)
        args = Namespace(**task["runner_args"])
        set_global_seed(int(task["metadata"]["random_seed"]))
        device = torch.device(args.device)
        field_calculator = make_field_calculator(args)
        write_tacview = bool(args.tacview) and get_worker_slot(int(args.num_workers)) == 1
        threat_target, attack_target, rollout_steps = rollout_counterfactual_label(
            metadata=task["metadata"],
            target_t=task["time_index"],
            agent_index=task["agent_index"],
            counterfactual_actions=np.asarray(task["counterfactual_actions"], dtype=np.int64),
            field_calculator=field_calculator,
            args=args,
            device=device,
            restore_ref=task.get("restore_ref"),
            factual_actions=np.asarray(task["factual_actions"], dtype=np.int64),
            factual_tacview_path=task.get("factual_tacview_path", "") if write_tacview else "",
            counterfactual_tacview_path=task.get("counterfactual_tacview_path", "") if write_tacview else "",
        )
        return {
            "status": "ok",
            "cf_name": task["cf_name"],
            "all_target_index": task["all_target_index"],
            "raw_file_index": task["raw_file_index"],
            "time_index": task["time_index"],
            "episode_id": task["episode_id"],
            "agent_index": task["agent_index"],
            "sample_category": task["sample_category"],
            "factual_actions": np.asarray(task["factual_actions"], dtype=np.int64),
            "counterfactual_actions": np.asarray(task["counterfactual_actions"], dtype=np.int64),
            "counterfactual_agent_action": np.asarray(task["counterfactual_agent_action"], dtype=np.int64),
            "threat_target": threat_target,
            "attack_target": attack_target,
            "rollout_steps": rollout_steps,
        }
    except BaseException as exc:
        return {
            "status": "failed",
            "cf_name": task.get("cf_name", ""),
            "all_target_index": task.get("all_target_index", -1),
            "raw_file": task.get("raw_file", ""),
            "time_index": task.get("time_index", -1),
            "agent_index": task.get("agent_index", -1),
            "error": repr(exc),
            "traceback": traceback.format_exc(),
        }


def iter_task_results(tasks, num_workers, chunk_size):
    if num_workers <= 1:
        for task in tasks:
            yield run_counterfactual_task(task)
        return

    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=int(num_workers)) as pool:
        yield from pool.imap_unordered(
            run_counterfactual_task,
            tasks,
            chunksize=max(1, int(chunk_size)),
        )


def row_category_name(category_id):
    category_id = int(category_id)
    if 0 <= category_id < len(CATEGORY_NAMES):
        return CATEGORY_NAMES[category_id]
    return str(category_id)


def empty_group_dataset(cf_name, split_name):
    return {
        "split_name": np.asarray(split_name, dtype=object),
        "counterfactual_name": np.asarray(cf_name, dtype=object),
        "all_target_indices": np.asarray([], dtype=np.int64),
        "raw_file_indices": np.asarray([], dtype=np.int64),
        "time_indices": np.asarray([], dtype=np.int32),
        "episode_ids_per_step": np.asarray([], dtype=np.int32),
        "agent_indices": np.asarray([], dtype=np.int16),
        "sample_category": np.asarray([], dtype=np.int16),
        "factual_actions": np.asarray([], dtype=np.int64).reshape(0, 0, 0),
        "counterfactual_actions": np.asarray([], dtype=np.int64).reshape(0, 0, 0),
        "counterfactual_agent_actions": np.asarray([], dtype=np.int64).reshape(0, 0),
        "threat_targets": np.asarray([], dtype=np.float32).reshape(0, 1),
        "attack_targets": np.asarray([], dtype=np.float32).reshape(0, 1),
        "rollout_steps": np.asarray([], dtype=np.int32),
        "source_files": np.asarray([], dtype=object),
        "sample_category_names": np.asarray(CATEGORY_NAMES, dtype=object),
    }


def pack_group_dataset(records, cf_name, split_name, source_files):
    if not records:
        return empty_group_dataset(cf_name, split_name)
    return {
        "split_name": np.asarray(split_name, dtype=object),
        "counterfactual_name": np.asarray(cf_name, dtype=object),
        "all_target_indices": np.asarray([r["all_target_index"] for r in records], dtype=np.int64),
        "raw_file_indices": np.asarray([r["raw_file_index"] for r in records], dtype=np.int64),
        "time_indices": np.asarray([r["time_index"] for r in records], dtype=np.int32),
        "episode_ids_per_step": np.asarray([r["episode_id"] for r in records], dtype=np.int32),
        "agent_indices": np.asarray([r["agent_index"] for r in records], dtype=np.int16),
        "sample_category": np.asarray([r["sample_category"] for r in records], dtype=np.int16),
        "factual_actions": np.asarray([r["factual_actions"] for r in records], dtype=np.int64),
        "counterfactual_actions": np.asarray([r["counterfactual_actions"] for r in records], dtype=np.int64),
        "counterfactual_agent_actions": np.asarray([r["counterfactual_agent_action"] for r in records], dtype=np.int64),
        "threat_targets": np.asarray([r["threat_target"] for r in records], dtype=np.float32).reshape(-1, 1),
        "attack_targets": np.asarray([r["attack_target"] for r in records], dtype=np.float32).reshape(-1, 1),
        "rollout_steps": np.asarray([r["rollout_steps"] for r in records], dtype=np.int32),
        "source_files": source_files,
        "sample_category_names": np.asarray(CATEGORY_NAMES, dtype=object),
    }


def save_npz(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)


def save_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def build_split(split_name, split_path, output_dir, args):
    data_dir = split_path.parent
    split_indices, all_target = read_all_target(data_dir, split_path)
    if args.max_points > 0:
        split_indices = split_indices[: int(args.max_points)]

    raw_cache = {}
    restore_lookup = load_restore_lookup(args.restore_dir) if args.restore_dir else {}
    tasks = []
    prepare_failures = []
    runner_args = {
        "num_agents_total": int(args.num_agents_total),
        "deterministic": bool(args.deterministic),
        "fix_position": bool(args.fix_position),
        "device": str(args.device),
        "num_workers": int(args.num_workers),
        "tacview": bool(args.tacview),
        "recurrent_hidden_size_actor": int(args.recurrent_hidden_size_actor),
        "field_k_step": int(args.field_k_step),
        "field_gamma": float(args.field_gamma),
        "ego_team": float(args.ego_team),
        "r_min": float(args.r_min),
        "r_attack": float(args.r_attack),
        "r_nez": float(args.r_nez),
        "theta_attack_deg": float(args.theta_attack_deg),
        "theta_nez_deg": float(args.theta_nez_deg),
    }

    logging.info(f"[split:{split_name}] points={len(split_indices)}")
    for point_i, row in enumerate(split_indices, start=1):
        row = int(row)
        raw_file_index = int(all_target["raw_file_indices"][row])
        raw_path = resolve_project_path(all_target["source_files"][raw_file_index])
        time_index = int(all_target["time_indices"][row])

        try:
            if raw_file_index not in raw_cache:
                raw_cache[raw_file_index] = load_raw_metadata(raw_path)
            raw_ego_actions, raw_all_actions, metadata = raw_cache[raw_file_index]

            if time_index < 0 or time_index >= raw_ego_actions.shape[0]:
                raise IndexError(f"bad time_index={time_index} for {raw_path}")

            factual_ego_actions = raw_ego_actions[time_index].astype(np.int64, copy=True)
            previous_ego_actions = (
                raw_ego_actions[time_index - 1].astype(np.int64, copy=False)
                if time_index > 0
                else factual_ego_actions
            )
            factual_actions = (
                raw_all_actions[time_index].astype(np.int64, copy=True)
                if raw_all_actions is not None
                else factual_ego_actions.copy()
            )

            num_ego_agents = int(args.num_agents_total) // 2
            if factual_ego_actions.shape[0] != num_ego_agents:
                raise ValueError(
                    f"{raw_path}: raw ego action count={factual_ego_actions.shape[0]}, expected={num_ego_agents}"
                )
            if raw_all_actions is not None and factual_actions.shape[0] != int(args.num_agents_total):
                raise ValueError(
                    f"{raw_path}: raw all action count={factual_actions.shape[0]}, expected={args.num_agents_total}"
                )
            if args.restore_dir and row not in restore_lookup:
                raise KeyError(
                    f"restore state for all_target_index={row} was not found under {args.restore_dir}. "
                    "Run collect_restore_states.py for this split first."
                )

            for cf_name in CF_ACTION_NAMES:
                for agent_index in range(num_ego_agents):
                    cf_actions = factual_actions.copy()
                    cf_actions[agent_index] = counterfactual_action(
                        cf_name,
                        factual_ego_actions[agent_index],
                        previous_ego_actions[agent_index],
                    )
                    tasks.append(
                        {
                            "runner_args": runner_args,
                            "metadata": metadata,
                            "cf_name": cf_name,
                            "all_target_index": row,
                            "raw_file_index": raw_file_index,
                            "raw_file": normalize_path(raw_path),
                            "time_index": time_index,
                            "episode_id": int(all_target["episode_ids_per_step"][row]),
                            "agent_index": agent_index,
                            "sample_category": int(all_target["sample_category"][row]),
                            "factual_actions": factual_actions,
                            "counterfactual_actions": cf_actions,
                            "counterfactual_agent_action": cf_actions[agent_index],
                            "restore_ref": restore_lookup.get(row),
                            "factual_tacview_path": normalize_path(
                                args.tacview_dir / f"{split_name}_row{row:08d}_t{time_index:06d}_factual.txt.acmi"
                            ),
                            "counterfactual_tacview_path": normalize_path(
                                args.tacview_dir
                                / f"{split_name}_row{row:08d}_t{time_index:06d}_{cf_name}_agent{agent_index:02d}.txt.acmi"
                            ),
                        }
                    )

            if point_i % max(1, int(args.log_interval)) == 0:
                logging.info(f"  prepared [{point_i}/{len(split_indices)}] row={row} t={time_index}")

        except Exception as exc:
            prepare_failures.append(
                {
                    "all_target_index": row,
                    "raw_file": normalize_path(raw_path),
                    "time_index": time_index,
                    "error": repr(exc),
                }
            )
            logging.info(f"  prepare failed [{point_i}/{len(split_indices)}] row={row} t={time_index}: {repr(exc)}")

    records_by_cf = {name: [] for name in CF_ACTION_NAMES}
    rollout_failures = []
    logging.info(
        f"[split:{split_name}] prepared_tasks={len(tasks)} "
        f"| workers={args.num_workers} | chunk_size={args.task_chunk_size}"
    )
    if args.tacview:
        logging.info(f"[split:{split_name}] tacview enabled: {normalize_path(args.tacview_dir)}")

    for done_i, result in enumerate(
        iter_task_results(tasks, int(args.num_workers), int(args.task_chunk_size)),
        start=1,
    ):
        if result.get("status") == "ok":
            records_by_cf[result["cf_name"]].append(result)
        else:
            rollout_failures.append(result)

        if done_i % max(1, int(args.log_interval)) == 0 or done_i == len(tasks):
            ok_count = done_i - len(rollout_failures)
            logging.info(f"  rollout [{done_i}/{len(tasks)}] ok={ok_count} failed={len(rollout_failures)}")

    split_outputs = {}
    for cf_name, records in records_by_cf.items():
        records = sorted(records, key=lambda r: (int(r["all_target_index"]), int(r["agent_index"])))
        payload = pack_group_dataset(records, cf_name, split_name, all_target["source_files"])
        out_path = output_dir / f"{split_name}_{cf_name}.npz"
        save_npz(out_path, payload)
        split_outputs[cf_name] = {
            "file": out_path.name,
            "rows": int(len(records)),
        }
        logging.info(f"[split:{split_name}:{cf_name}] saved {normalize_path(out_path)} rows={len(records)}")

    return {
        "split": split_name,
        "input": normalize_path(split_path),
        "points": int(len(split_indices)),
        "tasks": int(len(tasks)),
        "outputs": {key: {"file": value["file"], "rows": value["rows"]} for key, value in split_outputs.items()},
        "prepare_failures": prepare_failures,
        "rollout_failures": rollout_failures,
    }


def get_parser():
    parser = argparse.ArgumentParser(description="Build COMA-style AeroTAF counterfactual val/test target datasets.")
    parser.add_argument("--dataset-dir", type=str, required=True, help="Processed dataset directory containing val/test/all_target npz files.")
    parser.add_argument("--output-dir", type=str, default="", help="Output directory. Defaults to dataset_dir/coma_counterfactual_K{K}.")
    parser.add_argument("--splits", type=str, default="val test", help="Space-separated splits to process.")
    parser.add_argument("--num-agents-total", type=int, default=8, help="Total aircraft count; ego count is total/2.")
    parser.add_argument("--deterministic", action="store_true", default=True, help="Use deterministic actor actions during replay/rollout.")
    parser.add_argument("--stochastic", action="store_false", dest="deterministic", help="Use stochastic actor actions.")
    parser.add_argument("--fix-position", action="store_true", default=False, help="Pass fix_position=True to MultipleCombatEnv.")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device for actors.")
    parser.add_argument("--recurrent-hidden-size-actor", type=int, default=128, help="Actor recurrent hidden size.")
    parser.add_argument("--max-points", type=int, default=0, help="Optional cap per split for debugging.")
    parser.add_argument("--log-interval", type=int, default=10, help="Progress log interval in source points.")
    parser.add_argument("--num-workers", type=int, default=1, help="Parallel worker process count. Use 1 for serial execution.")
    parser.add_argument("--task-chunk-size", type=int, default=1, help="Multiprocessing imap_unordered chunk size.")
    parser.add_argument("--restore-dir", type=str, default="", help="Optional restore_states directory from collect_restore_states.py.")
    parser.add_argument("--tacview", action="store_true", default=False, help="Write txt.acmi rollouts for Tacview visualization.")
    parser.add_argument("--tacview-dir", type=str, default="", help="Tacview output directory. Defaults to output_dir/tacview.")

    parser.add_argument("--field-k-step", type=int, default=100, help="Future horizon K for counterfactual target calculation.")
    parser.add_argument("--field-gamma", type=float, default=0.96, help="Discount gamma for K-step target calculation.")
    parser.add_argument("--ego-team", type=float, default=0.0, help="Ego team id used by FieldCalculator.")
    parser.add_argument("--r-min", type=float, default=4000.0, help="Minimum effective attack range.")
    parser.add_argument("--r-attack", type=float, default=14000.0, help="Attack-zone range threshold.")
    parser.add_argument("--r-nez", type=float, default=10000.0, help="No-escape-zone range threshold.")
    parser.add_argument("--theta-attack-deg", type=float, default=60.0, help="Attack-zone angle threshold in degrees.")
    parser.add_argument("--theta-nez-deg", type=float, default=30.0, help="No-escape-zone angle threshold in degrees.")
    return parser


def main(args):
    parser = get_parser()
    all_args = parser.parse_args(args)

    if all_args.num_agents_total % 2 != 0:
        raise ValueError("--num-agents-total must be even")

    dataset_dir = resolve_project_path(all_args.dataset_dir)
    if not dataset_dir.exists():
        raise FileNotFoundError(f"dataset dir not found: {dataset_dir}")

    output_dir = (
        resolve_project_path(all_args.output_dir)
        if all_args.output_dir
        else dataset_dir / f"coma_counterfactual_K{all_args.field_k_step}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    all_args.tacview_dir = (
        resolve_project_path(all_args.tacview_dir)
        if all_args.tacview_dir
        else output_dir / "tacview"
    )
    if all_args.tacview:
        all_args.tacview_dir.mkdir(parents=True, exist_ok=True)

    splits = [name.strip() for name in all_args.splits.split() if name.strip()]

    logging.info("=" * 72)
    logging.info("AeroTAF COMA Counterfactual Target Builder")
    logging.info("=" * 72)
    logging.info(f"dataset dir : {normalize_path(dataset_dir)}")
    logging.info(f"output dir  : {normalize_path(output_dir)}")
    logging.info(f"splits      : {splits}")
    logging.info(f"cf actions  : {list(CF_ACTION_NAMES)}")
    logging.info(f"K/gamma     : {all_args.field_k_step}/{all_args.field_gamma}")
    logging.info(f"workers     : {all_args.num_workers}")
    logging.info(f"restore     : {normalize_path(resolve_project_path(all_args.restore_dir)) if all_args.restore_dir else 'disabled, replay from seed/model to t'}")
    logging.info(f"tacview     : {normalize_path(all_args.tacview_dir) if all_args.tacview else 'disabled'}")
    logging.info("-" * 72)

    summaries = []
    for split_name in splits:
        split_path = dataset_dir / f"{split_name}.npz"
        if not split_path.exists():
            raise FileNotFoundError(f"split not found: {split_path}")
        summaries.append(build_split(split_name, split_path, output_dir, all_args))

    manifest = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_dir": normalize_path(dataset_dir),
        "output_dir": normalize_path(output_dir),
        "restore_dir": normalize_path(resolve_project_path(all_args.restore_dir)) if all_args.restore_dir else "",
        "tacview": {
            "enabled": bool(all_args.tacview),
            "output_dir": normalize_path(all_args.tacview_dir) if all_args.tacview else "",
            "mode": "txt.acmi",
            "scope": "all serial tasks, or tasks executed by worker slot 1 when num_workers > 1",
            "contains": "factual K-step rollout from t and counterfactual K-step rollouts from t",
        },
        "splits": splits,
        "counterfactual_actions": list(CF_ACTION_NAMES),
        "counterfactual_definition": {
            "unit": "single ego agent",
            "other_ego_agents": "keep factual action at t",
            "enemy_agents": "policy action at t and future steps",
            "future_policy": "ego/enemy actors continue closed-loop after the intervention step",
            "shoot_dimension": "always kept equal to factual shoot action for all counterfactual kinds",
        },
        "field_params": {
            "field_k_step": all_args.field_k_step,
            "field_gamma": all_args.field_gamma,
            "ego_team": all_args.ego_team,
            "r_min": all_args.r_min,
            "r_attack": all_args.r_attack,
            "r_nez": all_args.r_nez,
            "theta_attack_deg": all_args.theta_attack_deg,
            "theta_nez_deg": all_args.theta_nez_deg,
        },
        "summaries": summaries,
    }
    manifest_path = output_dir / "coma_counterfactual_manifest.json"
    save_json(manifest_path, manifest)
    logging.info(f"Saved manifest: {normalize_path(manifest_path)}")
    logging.info("Done.")


if __name__ == "__main__":
    default_args = [
        "--dataset-dir", "datasets/aerotaf/4v4_shoot_mappo_pool/fkr-300vs500/processed_detail_index_k_target_K50",
        "--restore-dir", "datasets/aerotaf/4v4_shoot_mappo_pool/fkr-300vs500/processed_detail_index_k_target_K50/restore_states",
        "--splits", "val test",
        "--field-k-step", "50",
        "--field-gamma", "0.96",
        "--num-agents-total", "8",
        "--device", "cpu",
        # "--tacview",
        "--num-workers", "64",
        "--task-chunk-size", "1",
        "--log-interval", "10",
    ]
    main(sys.argv[1:] if len(sys.argv) > 1 else default_args)
