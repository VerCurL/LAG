import os
import csv
import time
import logging
import traceback
import multiprocessing as mp
import argparse
from collections import defaultdict

import numpy as np
import torch
import matplotlib.pyplot as plt

from envs.JSBSim.envs import MultipleCombatEnv
from algorithms.mappo.ppo_actor import PPOActor as MAPPOActor
from algorithms.mappoCFC.ppo_actor import PPOActor as MAPPOCFCActor

def _parent_tag(path):
    path = os.path.normpath(path)
    parent = os.path.basename(os.path.dirname(path))
    return parent if parent else os.path.basename(path)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_games", type=int, required=True)
    parser.add_argument("--max_parallel", type=int, required=True)
    parser.add_argument("--ego_policy_index", required=True)
    parser.add_argument("--enm_policy_index", required=True)
    parser.add_argument("--ego_run_dir", required=True)
    parser.add_argument("--enm_run_dir", required=True)
    return parser.parse_args()

class Args_EGO:
    def __init__(self) -> None:
        self.gain = 0.01
        self.hidden_size_actor = '128 128'
        self.hidden_size_critic = '512 512'
        self.act_hidden_size_actor = '128 128'
        self.act_hidden_size_critic = '512 512'
        self.activation_id = 1
        self.use_feature_normalization = False
        self.use_recurrent_policy = True
        self.recurrent_hidden_size_actor = 128
        self.recurrent_hidden_size_critic = 512

        self.KQ_hidden_size = '128 128'
        self.V_hidden_size = '128 128'
        self.AeroTAF_out_hidden_size = '128'
        self.num_heads = 4
        self.AeroTAF_kstep = 20

        self.recurrent_hidden_layers = 1
        self.tpdv = dict(dtype=torch.float32, device=torch.device('cpu'))
        self.use_prior = True


class Args_ENM:
    def __init__(self) -> None:
        self.gain = 0.01
        self.hidden_size_actor = '128 128'
        self.hidden_size_critic = '512 512'
        self.act_hidden_size_actor = '128 128'
        self.act_hidden_size_critic = '512 512'
        self.activation_id = 1
        self.use_feature_normalization = False
        self.use_recurrent_policy = True
        self.recurrent_hidden_size_actor = 128
        self.recurrent_hidden_size_critic = 512
        self.recurrent_hidden_layers = 1
        self.tpdv = dict(dtype=torch.float32, device=torch.device('cpu'))
        self.use_prior = True


class EpisodeMetricCollector:
    """
    只在内存中统计单局对战 summary，不写每局 step_metrics，也不创建 evaluation/episode_x。
    """

    def __init__(self, ego_agent_ids, enm_agent_ids):
        self.ego_agent_ids = list(ego_agent_ids)
        self.enm_agent_ids = list(enm_agent_ids)
        self._missile_owner = {}
        self._missile_last_status = {}
        self._current_episode_steps = []
        self._current_episode_index = None

    def start_episode(self, episode_index):
        self._current_episode_index = int(episode_index)
        self._current_episode_steps = []
        self._missile_owner = {}
        self._missile_last_status = {}

    def record_step(self, env, info, step_index):
        reward_breakdown = {}
        flight_metrics = {}
        if isinstance(info, dict):
            reward_breakdown = info.get("reward_breakdown", {})
            flight_metrics = info.get("flight_metrics", {})

        missile_stats = self._update_missile_stats(env)
        step_row = {
            "episode": self._current_episode_index,
            "step": int(step_index),
        }

        for team_name, agent_ids in (("ego", self.ego_agent_ids), ("enm", self.enm_agent_ids)):
            team_metrics = self._collect_team_metrics(
                env=env,
                agent_ids=agent_ids,
                reward_breakdown=reward_breakdown,
                flight_metrics=flight_metrics,
            )
            for key, value in team_metrics.items():
                step_row[f"{team_name}_{key}"] = value
            for key, value in missile_stats[team_name].items():
                step_row[f"{team_name}_{key}"] = value

        step_row["reward_gap"] = step_row["ego_reward_step_sum"] - step_row["enm_reward_step_sum"]
        step_row["blood_gap"] = step_row["ego_total_blood"] - step_row["enm_total_blood"]
        self._current_episode_steps.append(step_row)

    def finalize_episode(self):
        if not self._current_episode_steps:
            return None

        final_row = dict(self._current_episode_steps[-1])
        return {
            "episode": self._current_episode_index,
            "steps": final_row["step"],
            "winner": self._determine_winner(final_row),
            "ego_alive_count": final_row["ego_alive_count"],
            "enm_alive_count": final_row["enm_alive_count"],
            "ego_total_blood": final_row["ego_total_blood"],
            "enm_total_blood": final_row["enm_total_blood"],
            "ego_reward_sum": final_row["ego_cumulative_reward"],
            "enm_reward_sum": final_row["enm_cumulative_reward"],
            "ego_missiles_launched": final_row["ego_missiles_launched"],
            "enm_missiles_launched": final_row["enm_missiles_launched"],
            "ego_missile_hits": final_row["ego_missile_hits"],
            "enm_missile_hits": final_row["enm_missile_hits"],
            "ego_missile_misses": final_row["ego_missile_misses"],
            "enm_missile_misses": final_row["enm_missile_misses"],
            "ego_hit_rate": self._safe_ratio(final_row["ego_missile_hits"], final_row["ego_missiles_launched"]),
            "enm_hit_rate": self._safe_ratio(final_row["enm_missile_hits"], final_row["enm_missiles_launched"]),
            "ego_avg_nearest_enemy_distance_m": self._mean_of_column("ego_avg_nearest_enemy_distance_m"),
            "enm_avg_nearest_enemy_distance_m": self._mean_of_column("enm_avg_nearest_enemy_distance_m"),
            "ego_avg_altitude_m": self._mean_of_column("ego_avg_altitude_m"),
            "enm_avg_altitude_m": self._mean_of_column("enm_avg_altitude_m"),
        }

    def _collect_team_metrics(self, env, agent_ids, reward_breakdown, flight_metrics):
        alive_count = 0
        shotdown_count = 0
        crash_count = 0
        total_blood = 0.0
        reward_step_sum = 0.0
        altitudes = []
        headings = []
        nearest_enemy_distances = []

        for agent_id in agent_ids:
            agent = env.agents[agent_id]
            if agent.is_alive:
                alive_count += 1
            if agent.is_shotdown:
                shotdown_count += 1
            if agent.is_crash:
                crash_count += 1

            total_blood += float(agent.bloods)
            reward_step_sum += float(reward_breakdown.get(agent_id, {}).get("total", 0.0))

            metrics = flight_metrics.get(agent_id, {})
            altitudes.append(metrics.get("altitude_m", np.nan))
            headings.append(metrics.get("heading_deg", np.nan))
            nearest_enemy_distances.append(metrics.get("nearest_enemy_distance_m", np.nan))

        previous_cum = 0.0
        if self._current_episode_steps:
            team_key = "ego" if agent_ids == self.ego_agent_ids else "enm"
            previous_cum = float(self._current_episode_steps[-1][f"{team_key}_cumulative_reward"])

        return {
            "alive_count": alive_count,
            "shotdown_count": shotdown_count,
            "crash_count": crash_count,
            "total_blood": total_blood,
            "reward_step_sum": reward_step_sum,
            "cumulative_reward": previous_cum + reward_step_sum,
            "avg_altitude_m": self._nanmean(altitudes),
            "avg_heading_deg": self._nanmean(headings),
            "avg_nearest_enemy_distance_m": self._nanmean(nearest_enemy_distances),
        }

    def _update_missile_stats(self, env):
        stats = {
            "ego": {"missiles_launched": 0, "missile_hits": 0, "missile_misses": 0, "active_missiles": 0},
            "enm": {"missiles_launched": 0, "missile_hits": 0, "missile_misses": 0, "active_missiles": 0},
        }

        if self._current_episode_steps:
            for team_name in stats.keys():
                for key in ("missiles_launched", "missile_hits", "missile_misses"):
                    stats[team_name][key] = int(self._current_episode_steps[-1][f"{team_name}_{key}"])

        for team_name, agent_ids in (("ego", self.ego_agent_ids), ("enm", self.enm_agent_ids)):
            for agent_id in agent_ids:
                for missile in env.agents[agent_id].launch_missiles:
                    missile_uid = missile.uid
                    current_status = int(missile.m_status)

                    if missile_uid not in self._missile_owner:
                        self._missile_owner[missile_uid] = team_name
                        stats[team_name]["missiles_launched"] += 1

                    if missile.is_alive:
                        stats[team_name]["active_missiles"] += 1

                    last_status = self._missile_last_status.get(missile_uid)
                    if current_status != last_status:
                        if missile.is_success:
                            stats[team_name]["missile_hits"] += 1
                        elif missile.is_done:
                            stats[team_name]["missile_misses"] += 1
                        self._missile_last_status[missile_uid] = current_status

        return stats

    def _determine_winner(self, final_row):
        if final_row["ego_alive_count"] > final_row["enm_alive_count"]:
            return "ego"
        if final_row["ego_alive_count"] < final_row["enm_alive_count"]:
            return "enm"
        return "draw"

    def _mean_of_column(self, key):
        return self._nanmean([row.get(key, np.nan) for row in self._current_episode_steps])

    @staticmethod
    def _nanmean(values):
        if not values:
            return np.nan
        array = np.asarray(values, dtype=float)
        if np.all(np.isnan(array)):
            return np.nan
        return float(np.nanmean(array))

    @staticmethod
    def _safe_ratio(numerator, denominator):
        if denominator <= 0:
            return 0.0
        return float(numerator) / float(denominator)


def _t2n(x):
    return x.detach().cpu().numpy()


def _load_state_dict(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def configure_worker_logging(worker_rank: int, debug_first_worker: bool):
    if debug_first_worker and worker_rank == 0:
        level = logging.DEBUG
    else:
        level = logging.WARNING

    logging.basicConfig(
        level=level,
        format="%(levelname)s:%(name)s:%(message)s",
        force=True,
    )
    logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)


def run_one_episode(task):
    episode_id = task["episode_id"]
    worker_rank = task["worker_rank"]
    base_seed = task["base_seed"]
    file_path = task["file_path"]
    render = task["render"]

    configure_worker_logging(worker_rank, task["debug_first_worker"])

    torch.set_num_threads(1)

    device = torch.device(task["device"])
    num_agents = task["num_agents"]

    env = None
    try:
        env = MultipleCombatEnv(
            config_name=task["scenario_name"],
            policy_type=task["policy_type"],
            fix_position=task["fix_position"],
        )
        env.seed(base_seed + episode_id * 1000)

        args_ego = Args_EGO()
        args_enm = Args_ENM()
        args_ego.num_agents = args_enm.num_agents = num_agents // 2

        ego_policy = MAPPOCFCActor(args_ego, env.observation_space, env.action_space, device=device)
        enm_policy = MAPPOActor(args_enm, env.observation_space, env.action_space, device=device)

        ego_policy.eval()
        enm_policy.eval()

        ego_policy.load_state_dict(_load_state_dict(task["ego_model_path"], device))
        enm_policy.load_state_dict(_load_state_dict(task["enm_model_path"], device))

        collector = EpisodeMetricCollector(
            ego_agent_ids=env.ego_ids,
            enm_agent_ids=env.enm_ids,
        )

        obs, _ = env.reset()

        acmi_path = os.path.join(file_path, f"Combat_{episode_id}.txt.acmi")
        if render:
            env.render(mode="txt", filepath=acmi_path)

        collector.start_episode(episode_id)

        masks = np.ones((num_agents // 2, 1), dtype=np.float32)
        ego_rnn_states = np.zeros((1, 1, 128), dtype=np.float32)
        enm_rnn_states = np.zeros_like(ego_rnn_states, dtype=np.float32)

        ego_obs = obs[:num_agents // 2, :]
        enm_obs = obs[num_agents // 2:, :]

        episode_rewards = 0.0
        time_net = 0.0
        time_env = 0.0

        while True:
            start_net = time.time()
            with torch.no_grad():
                ego_actions, _, ego_rnn_states = ego_policy(
                    ego_obs, ego_rnn_states, masks, deterministic=True
                )
                enm_actions, _, enm_rnn_states = enm_policy(
                    enm_obs, enm_rnn_states, masks, deterministic=True
                )

            ego_actions = _t2n(ego_actions)
            enm_actions = _t2n(enm_actions)
            ego_rnn_states = _t2n(ego_rnn_states)
            enm_rnn_states = _t2n(enm_rnn_states)

            actions = np.concatenate((ego_actions, enm_actions), axis=0)
            time_net += time.time() - start_net

            start_env = time.time()
            obs, _, rewards, dones, infos = env.step(actions)
            time_env += time.time() - start_env

            ego_rewards = rewards[:num_agents // 2, ...]
            episode_rewards += float(np.sum(ego_rewards))

            collector.record_step(
                env=env,
                info=infos,
                step_index=env.current_step,
            )

            if render:
                env.render(mode="txt", filepath=acmi_path)

            if np.all(dones):
                break

            ego_obs = obs[:num_agents // 2, :]
            enm_obs = obs[num_agents // 2:, :]

        summary = collector.finalize_episode()
        env.refresh_records()

        if summary is None:
            summary = {}

        summary = dict(summary)
        summary["episode"] = episode_id
        summary["episode_reward"] = episode_rewards
        summary["time_net"] = time_net
        summary["time_env"] = time_env
        summary["acmi_path"] = acmi_path
        summary["status"] = "ok"

        return summary

    except BaseException as exc:
        return {
            "episode": episode_id,
            "status": "failed",
            "error": repr(exc),
            "traceback": traceback.format_exc(),
        }

    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass


def write_episode_metrics(rows, csv_path):
    rows = sorted(rows, key=lambda x: int(x.get("episode", 0)))
    if not rows:
        return

    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_summary(rows, save_dir):
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    ok_rows = sorted(ok_rows, key=lambda x: int(x.get("episode", 0)))
    if not ok_rows:
        return

    os.makedirs(save_dir, exist_ok=True)

    episodes = [int(row["episode"]) for row in ok_rows]
    winners = [row.get("winner", "unknown") for row in ok_rows]
    win_counts = {
        "ego": winners.count("ego"),
        "enm": winners.count("enm"),
        "draw": winners.count("draw"),
        "unknown": winners.count("unknown"),
    }

    plt.figure(figsize=(8, 5))
    plt.bar(["ego", "enm", "draw", "unknown"], [
        win_counts["ego"],
        win_counts["enm"],
        win_counts["draw"],
        win_counts["unknown"],
    ])
    plt.title("Win Count Summary")
    plt.ylabel("Count")
    plt.grid(axis="y", alpha=0.3)
    plt.savefig(os.path.join(save_dir, "summary_win_count.png"), dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(episodes, [float(row.get("ego_reward_sum", 0.0)) for row in ok_rows], marker="o", label="Ego Reward")
    plt.plot(episodes, [float(row.get("enm_reward_sum", 0.0)) for row in ok_rows], marker="o", label="Enemy Reward")
    plt.title("Episode Reward Summary")
    plt.xlabel("Episode")
    plt.ylabel("Reward Sum")
    plt.legend(loc="best")
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(save_dir, "summary_reward.png"), dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(episodes, [float(row.get("ego_hit_rate", 0.0)) for row in ok_rows], marker="o", label="Ego Hit Rate")
    plt.plot(episodes, [float(row.get("enm_hit_rate", 0.0)) for row in ok_rows], marker="o", label="Enemy Hit Rate")
    plt.title("Episode Missile Hit Rate")
    plt.xlabel("Episode")
    plt.ylabel("Hit Rate")
    plt.legend(loc="best")
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(save_dir, "summary_hit_rate.png"), dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(episodes, [float(row.get("ego_alive_count", 0.0)) for row in ok_rows], marker="o", label="Ego Alive")
    plt.plot(episodes, [float(row.get("enm_alive_count", 0.0)) for row in ok_rows], marker="o", label="Enemy Alive")
    plt.title("Episode Alive Count Summary")
    plt.xlabel("Episode")
    plt.ylabel("Alive Count")
    plt.legend(loc="best")
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(save_dir, "summary_alive_count.png"), dpi=150, bbox_inches="tight")
    plt.close()


def chunks(items, chunk_size):
    for start in range(0, len(items), chunk_size):
        yield items[start:start + chunk_size]


def main():
    args = parse_args()

    num_agents = 8
    num_games = args.num_games              # 总对战轮数
    max_parallel = args.max_parallel        # 最大并行计算数量
    render = True

    debug_first_worker = True

    # 并行进程多时，CUDA 会为每个进程各加载一份模型，显存压力会很大。
    # 如果你确认显存足够，可以改成 "cuda"。
    device = "cpu"
    # device = "cuda" if torch.cuda.is_available() else "cpu"

    ego_policy_index = args.ego_policy_index
    enm_policy_index = args.enm_policy_index

    ego_run_dir = args.ego_run_dir
    enm_run_dir = args.enm_run_dir

    experiment_name =  _parent_tag(ego_run_dir) + ".vs." + _parent_tag(enm_run_dir)
    experiment_file_name = (
        "../../gaming_result/MAPPOCFC.vs.MAPPO.parallel/"
        + experiment_name
        + "/["
        + str(ego_policy_index)
        + ","
        + str(enm_policy_index)
        + "]/"
    )

    file_path = os.path.join(
        experiment_file_name,
        time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()) + f"[{num_games}]",
    )
    os.makedirs(file_path, exist_ok=True)

    evaluation_dir = os.path.join(file_path, "evaluation")
    os.makedirs(evaluation_dir, exist_ok=True)

    base_task = {
        "num_agents": num_agents,
        "base_seed": 0,
        "file_path": file_path,
        "render": render,
        "device": device,
        "debug_first_worker": debug_first_worker,
        "scenario_name": "4v4/ShootMissile/HierarchySelfplay",
        "policy_type": "fkr",
        "fix_position": False,
        "ego_model_path": os.path.join(ego_run_dir, f"actor_{ego_policy_index}.pt"),
        "enm_model_path": os.path.join(enm_run_dir, f"actor_{enm_policy_index}.pt"),
    }

    all_results = []
    episodes = list(range(1, num_games + 1))

    for batch_id, batch_episodes in enumerate(chunks(episodes, max_parallel), start=1):
        print(f"======= batch {batch_id} begins: {batch_episodes} =======")

        tasks = []
        for worker_rank, episode_id in enumerate(batch_episodes):
            task = dict(base_task)
            task["episode_id"] = episode_id
            task["worker_rank"] = worker_rank
            tasks.append(task)

        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=len(tasks)) as pool:
            batch_results = pool.map(run_one_episode, tasks)

        all_results.extend(batch_results)

        episode_metrics_path = os.path.join(evaluation_dir, "episode_metrics.csv")
        write_episode_metrics(all_results, episode_metrics_path)
        plot_summary(all_results, evaluation_dir)

        for row in sorted(batch_results, key=lambda x: int(x.get("episode", 0))):
            if row.get("status") == "ok":
                print(
                    f"[episode {row['episode']}] "
                    f"winner={row.get('winner')} "
                    f"steps={row.get('steps')} "
                    f"reward={float(row.get('episode_reward', 0.0)):.3f}"
                )
            else:
                print(f"[episode {row['episode']}] failed: {row.get('error')}")

        print(f"======= batch {batch_id} ends =======")

    print(f"All results saved to: {file_path}")


if __name__ == "__main__":
    mp.freeze_support()
    main()