import multiprocessing as mp
import random
import traceback
from pathlib import Path

import numpy as np
import torch

from envs.JSBSim.envs import MultipleCombatEnv
from envs.JSBSim.situation.extractor import SituationExtractor
from algorithms.mappo.ppo_actor import PPOActor as MAPPOActor
from .path_utils import canonicalize_task_key, normalize_path, resolve_project_path


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


def _mean_team_metric(metric_source, team_ids):
    values = []
    for agent_id in team_ids:
        agent_metrics = metric_source.get(agent_id)
        if not agent_metrics:
            continue
        values.append(agent_metrics)

    if not values:
        return {}

    keys = set()
    for item in values:
        keys.update(item.keys())

    result = {}
    for key in keys:
        valid_values = []
        for item in values:
            value = item.get(key)
            if value is None:
                continue
            value = float(value)
            if np.isnan(value):
                continue
            valid_values.append(value)
        if valid_values:
            result[key] = float(sum(valid_values) / len(valid_values))
    return result


def _flatten_summary_metrics(metric_sums, metric_counts, prefix):
    flat = {}
    for key, value in metric_sums.items():
        count = metric_counts.get(key, 0)
        flat[f"{prefix}_{key}_mean"] = float(value / count) if count > 0 else 0.0
    return flat


def _append_metrics(metric_sums, metric_counts, metrics):
    for key, value in metrics.items():
        if value is None:
            continue
        value = float(value)
        if np.isnan(value):
            continue
        metric_sums[key] = metric_sums.get(key, 0.0) + value
        metric_counts[key] = metric_counts.get(key, 0) + 1


def run_episode_task(task):
    episode_id = task["episode_id"]
    seed = int(task["seed"])

    torch.set_num_threads(1)
    set_global_seed(seed)

    device = torch.device(task["device"])
    num_agents_total = task["num_agents_total"]
    num_ego_agents = num_agents_total // 2

    env = None
    try:
        env = MultipleCombatEnv(
            config_name=task["scenario_name"],
            policy_type=task["policy_type"],
            algorithm="mappo",
            fix_position=task["fix_position"],
        )
        if env.situation_extractor is None:
            env.situation_extractor = SituationExtractor()
        env.seed(seed)

        ego_actor = make_actor(env, task["ego_model_path"], device)
        enm_actor = make_actor(env, task["enm_model_path"], device)

        obs, _ = env.reset()
        ego_obs = obs[:num_ego_agents]
        enm_obs = obs[num_ego_agents:]

        ego_rnn_states = np.zeros(
            (num_ego_agents, 1, task["recurrent_hidden_size_actor"]),
            dtype=np.float32,
        )
        enm_rnn_states = np.zeros_like(ego_rnn_states)
        masks = np.ones((num_ego_agents, 1), dtype=np.float32)

        obs_list = []
        action_list = []
        ego_action_list = []
        enm_action_list = []
        all_action_list = []
        mask_list = [masks.copy()]
        snapshot_list = []
        done_list = []

        ego_flight_sums = {}
        ego_flight_counts = {}
        enm_flight_sums = {}
        enm_flight_counts = {}

        step = 0
        while True:
            with torch.no_grad():
                ego_actions, _, ego_rnn_states = ego_actor(
                    ego_obs,
                    ego_rnn_states,
                    masks,
                    deterministic=task["deterministic"],
                )
                enm_actions, _, enm_rnn_states = enm_actor(
                    enm_obs,
                    enm_rnn_states,
                    masks,
                    deterministic=task["deterministic"],
                )

            ego_actions = t2n(ego_actions)
            enm_actions = t2n(enm_actions)
            ego_rnn_states = t2n(ego_rnn_states)
            enm_rnn_states = t2n(enm_rnn_states)

            actions = np.concatenate((ego_actions, enm_actions), axis=0)
            obs_list.append(ego_obs.astype(np.float32, copy=True))
            action_list.append(ego_actions.astype(np.float32, copy=True))
            ego_action_list.append(ego_actions.astype(np.int64, copy=True))
            enm_action_list.append(enm_actions.astype(np.int64, copy=True))
            all_action_list.append(actions.astype(np.int64, copy=True))

            next_obs, _, _, dones, info = env.step(actions)

            snapshot = info.get("AeroTAF_snapshot")
            if snapshot is None:
                raise RuntimeError(
                    "AeroTAF_snapshot is missing. SituationExtractor must be attached during collection."
                )

            snapshot_list.append(snapshot)
            done_list.append(dones[:num_ego_agents].astype(np.float32, copy=True))

            flight_metrics = info.get("flight_metrics", {})

            ego_team_ids = list(env.ego_ids)
            enm_team_ids = list(env.enm_ids)

            ego_flight_metrics = _mean_team_metric(flight_metrics, ego_team_ids)
            enm_flight_metrics = _mean_team_metric(flight_metrics, enm_team_ids)

            _append_metrics(ego_flight_sums, ego_flight_counts, ego_flight_metrics)
            _append_metrics(enm_flight_sums, enm_flight_counts, enm_flight_metrics)

            step += 1
            done_env = bool(np.all(dones))

            next_masks = np.ones((num_ego_agents, 1), dtype=np.float32)
            if done_env:
                next_masks[:] = 0.0
            mask_list.append(next_masks.copy())

            if done_env or step >= task["max_episode_steps"]:
                break

            masks = next_masks
            ego_obs = next_obs[:num_ego_agents]
            enm_obs = next_obs[num_ego_agents:]

        ego_alive_count = sum(1 for agent_id in env.ego_ids if env.agents[agent_id].is_alive)
        enm_alive_count = sum(1 for agent_id in env.enm_ids if env.agents[agent_id].is_alive)
        ego_loss_count = len(env.ego_ids) - ego_alive_count
        enm_loss_count = len(env.enm_ids) - enm_alive_count

        if ego_alive_count > enm_alive_count:
            winner = "ego"
        elif ego_alive_count < enm_alive_count:
            winner = "enm"
        else:
            winner = "draw"

        alive_margin = ego_alive_count - enm_alive_count

        result = {
            "episode": episode_id,
            "status": "ok",
            "task_kind": task["task_kind"],
            "task_key": canonicalize_task_key(task["task_key"]),
            "seed": seed,
            "steps": step,
            "winner": winner,
            "alive_margin": int(alive_margin),
            "ego_alive_count": int(ego_alive_count),
            "enm_alive_count": int(enm_alive_count),
            "ego_loss_count": int(ego_loss_count),
            "enm_loss_count": int(enm_loss_count),
            "ego_survival_rate": float(ego_alive_count / max(len(env.ego_ids), 1)),
            "enm_survival_rate": float(enm_alive_count / max(len(env.enm_ids), 1)),
            "ego_model_path": normalize_path(task["ego_model_path"]),
            "enm_model_path": normalize_path(task["enm_model_path"]),
            "scenario_id": task["scenario_id"],
            "scenario_bucket": task["scenario_bucket"],
            "pair_type": task["pair_type"],
            "ego_level": task["ego_level"],
            "enm_level": task["enm_level"],
            "ego_style": task["ego_style"],
            "enm_style": task["enm_style"],
            "ego_stage_hint": task["ego_stage_hint"],
            "enm_stage_hint": task["enm_stage_hint"],
        }
        result.update(_flatten_summary_metrics(ego_flight_sums, ego_flight_counts, "ego"))
        result.update(_flatten_summary_metrics(enm_flight_sums, enm_flight_counts, "enm"))

        if task["save_raw"]:
            out_dir = resolve_project_path(task["out_dir"])
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"episode_{episode_id:06d}.npz"
            np.savez_compressed(
                out_path,
                obs=np.asarray(obs_list, dtype=np.float32),
                actions=np.asarray(action_list, dtype=np.float32),
                ego_actions=np.asarray(ego_action_list, dtype=np.int64),
                enm_actions=np.asarray(enm_action_list, dtype=np.int64),
                all_actions=np.asarray(all_action_list, dtype=np.int64),
                masks=np.asarray(mask_list, dtype=np.float32),
                dones=np.asarray(done_list, dtype=np.float32),
                snapshots=np.asarray(snapshot_list, dtype=object),
                episode_id=np.asarray(episode_id, dtype=np.int32),
                task_key=canonicalize_task_key(task["task_key"]),
                task_kind=task["task_kind"],
                random_seed=np.asarray(seed, dtype=np.int32),
                scenario_id=task["scenario_id"],
                scenario_bucket=task["scenario_bucket"],
                ego_model_path=normalize_path(task["ego_model_path"]),
                enm_model_path=normalize_path(task["enm_model_path"]),
                ego_level=task["ego_level"],
                enm_level=task["enm_level"],
                ego_style=task["ego_style"],
                enm_style=task["enm_style"],
                pair_type=task["pair_type"],
                ego_stage_hint=task["ego_stage_hint"],
                enm_stage_hint=task["enm_stage_hint"],
                scenario_name=task["scenario_name"],
                policy_type=task["policy_type"],
            )
            result["file"] = normalize_path(out_path)

        return result

    except BaseException as exc:
        return {
            "episode": episode_id,
            "status": "failed",
            "task_kind": task.get("task_kind", "unknown"),
            "task_key": canonicalize_task_key(task.get("task_key", "")),
            "ego_model_path": normalize_path(task.get("ego_model_path", "")) if task.get("ego_model_path") else "",
            "enm_model_path": normalize_path(task.get("enm_model_path", "")) if task.get("enm_model_path") else "",
            "scenario_id": task.get("scenario_id", ""),
            "seed": task.get("seed", -1),
            "error": repr(exc),
            "traceback": traceback.format_exc(),
        }

    finally:
        if env is not None:
            env.close()


def _chunks(items, chunk_size):
    for start in range(0, len(items), chunk_size):
        yield items[start:start + chunk_size]


def execute_tasks(tasks, max_parallel):
    if not tasks:
        return []

    results = []
    ctx = mp.get_context("spawn")

    for batch_id, batch in enumerate(_chunks(tasks, max_parallel), start=1):
        for worker_rank, task in enumerate(batch):
            task["worker_rank"] = worker_rank

        print(f"batch {batch_id}: tasks={len(batch)}")
        with ctx.Pool(processes=len(batch)) as pool:
            batch_results = pool.map(run_episode_task, batch)
        results.extend(batch_results)

        ok = sum(1 for row in results if row["status"] == "ok")
        failed = len(results) - ok
        print(f"completed={ok}, failed={failed}")

    return results
