import multiprocessing as mp
import traceback
from pathlib import Path

import numpy as np
import torch

from envs.JSBSim.envs import MultipleCombatEnv
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
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device(task["device"])
    num_agents_total = task["num_agents_total"]
    num_ego_agents = num_agents_total // 2

    env = None
    try:
        env = MultipleCombatEnv(
            config_name=task["scenario_name"],
            policy_type=task["policy_type"],
            algorithm="mappoCFC",
            fix_position=task["fix_position"],
        )
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
        mask_list = [masks.copy()]
        snapshot_list = []
        reward_list = []
        done_list = []

        ego_reward_total = 0.0
        enm_reward_total = 0.0
        ego_flight_sums = {}
        ego_flight_counts = {}
        ego_reward_breakdown_sums = {}
        ego_reward_breakdown_counts = {}

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

            next_obs, _, rewards, dones, info = env.step(actions)

            snapshot = info.get("AeroTAF_snapshot")
            if snapshot is None:
                raise RuntimeError(
                    "AeroTAF_snapshot is missing. Make sure env is created with algorithm='mappoCFC'."
                )

            snapshot_list.append(snapshot)
            reward_list.append(rewards[:num_ego_agents].astype(np.float32, copy=True))
            done_list.append(dones[:num_ego_agents].astype(np.float32, copy=True))

            ego_reward_step = float(np.mean(rewards[:num_ego_agents]))
            enm_reward_step = float(np.mean(rewards[num_ego_agents:]))
            ego_reward_total += ego_reward_step
            enm_reward_total += enm_reward_step

            reward_breakdown = info.get("reward_breakdown", {})
            flight_metrics = info.get("flight_metrics", {})

            ego_team_ids = list(env.ego_ids)
            enm_team_ids = list(env.enm_ids)

            ego_reward_metrics = _mean_team_metric(reward_breakdown, ego_team_ids)
            ego_flight_metrics = _mean_team_metric(flight_metrics, ego_team_ids)

            _append_metrics(ego_reward_breakdown_sums, ego_reward_breakdown_counts, ego_reward_metrics)
            _append_metrics(ego_flight_sums, ego_flight_counts, ego_flight_metrics)

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

        if ego_alive_count > enm_alive_count:
            winner = "ego"
        elif ego_alive_count < enm_alive_count:
            winner = "enm"
        elif ego_reward_total > enm_reward_total:
            winner = "ego"
        elif ego_reward_total < enm_reward_total:
            winner = "enm"
        else:
            winner = "draw"

        reward_margin = ego_reward_total - enm_reward_total
        alive_margin = ego_alive_count - enm_alive_count

        result = {
            "episode": episode_id,
            "status": "ok",
            "task_kind": task["task_kind"],
            "task_key": canonicalize_task_key(task["task_key"]),
            "seed": seed,
            "steps": step,
            "winner": winner,
            "reward_margin": float(reward_margin),
            "alive_margin": int(alive_margin),
            "ego_alive_count": int(ego_alive_count),
            "enm_alive_count": int(enm_alive_count),
            "ego_total_reward": float(ego_reward_total),
            "enm_total_reward": float(enm_reward_total),
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
        result.update(_flatten_summary_metrics(ego_reward_breakdown_sums, ego_reward_breakdown_counts, "ego_reward"))

        result["ego_speed_mps_mean"] = result.pop("ego_speed_mps_mean", result.get("ego_speed_mps_mean", 0.0))
        result["ego_nearest_enemy_distance_m_mean"] = result.pop(
            "ego_nearest_enemy_distance_m_mean",
            result.get("ego_nearest_enemy_distance_m_mean", 0.0),
        )
        result["ego_attack_window_reward_mean"] = result.pop(
            "ego_reward_FKR_4v4_AttackWindowReward_mean",
            result.get("ego_reward_FKR_4v4_AttackWindowReward_mean", 0.0),
        )
        result["ego_missile_avoid_reward_mean"] = result.pop(
            "ego_reward_FKR_4v4_MissileAvoidReward_mean",
            result.get("ego_reward_FKR_4v4_MissileAvoidReward_mean", 0.0),
        )

        if task["save_raw"]:
            out_dir = resolve_project_path(task["out_dir"])
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"episode_{episode_id:06d}.npz"
            np.savez_compressed(
                out_path,
                obs=np.asarray(obs_list, dtype=np.float32),
                actions=np.asarray(action_list, dtype=np.float32),
                masks=np.asarray(mask_list, dtype=np.float32),
                rewards=np.asarray(reward_list, dtype=np.float32),
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
