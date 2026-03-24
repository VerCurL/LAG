import csv
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np


class BattleEvaluator:
    """
    Collect 4v4 battle metrics, export CSV files, and generate plots.
    """

    def __init__(self, save_dir, ego_agent_ids, enm_agent_ids):
        self.save_dir = save_dir
        self.ego_agent_ids = list(ego_agent_ids)
        self.enm_agent_ids = list(enm_agent_ids)
        os.makedirs(self.save_dir, exist_ok=True)

        self.episode_rows = []
        self._missile_owner = {}
        self._missile_last_status = {}
        self._current_episode_steps = []
        self._current_episode_index = None

    def start_episode(self, episode_index):
        self._current_episode_index = int(episode_index)
        self._current_episode_steps = []
        self._missile_owner = {}
        self._missile_last_status = {}

    def record_step(self, env, info, rewards, step_index):
        if self._current_episode_index is None:
            raise RuntimeError("Call start_episode() before record_step().")

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

    def finalize_episode(self, env):
        if not self._current_episode_steps:
            return None

        episode_dir = os.path.join(self.save_dir, f"episode_{self._current_episode_index}")
        os.makedirs(episode_dir, exist_ok=True)

        step_csv_path = os.path.join(episode_dir, "step_metrics.csv")
        self._write_csv(step_csv_path, self._current_episode_steps)

        final_row = dict(self._current_episode_steps[-1])
        summary_row = {
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
        self.episode_rows.append(summary_row)
        self._write_csv(os.path.join(episode_dir, "episode_summary.csv"), [summary_row])
        self._plot_episode_curves(episode_dir)
        return summary_row

    def finalize_all(self):
        if not self.episode_rows:
            return
        summary_csv_path = os.path.join(self.save_dir, "episode_metrics.csv")
        self._write_csv(summary_csv_path, self.episode_rows)
        self._plot_summary_curves(self.save_dir)

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
        # if final_row["ego_total_blood"] > final_row["enm_total_blood"]:
        #     return "ego"
        # if final_row["ego_total_blood"] < final_row["enm_total_blood"]:
        #     return "enm"
        # if final_row["ego_cumulative_reward"] > final_row["enm_cumulative_reward"]:
        #     return "ego"
        # if final_row["ego_cumulative_reward"] < final_row["enm_cumulative_reward"]:
        #     return "enm"
        return "draw"

    def _plot_episode_curves(self, episode_dir):
        self._plot_two_team_line(
            x_key="step",
            y_keys=("ego_total_blood", "enm_total_blood"),
            labels=("Ego Total Blood", "Enemy Total Blood"),
            title="Total Blood Over Time",
            ylabel="Blood",
            save_path=os.path.join(episode_dir, "blood_curve.png"),
        )
        self._plot_two_team_line(
            x_key="step",
            y_keys=("ego_alive_count", "enm_alive_count"),
            labels=("Ego Alive Count", "Enemy Alive Count"),
            title="Alive Count Over Time",
            ylabel="Alive Count",
            save_path=os.path.join(episode_dir, "alive_curve.png"),
        )
        self._plot_two_team_line(
            x_key="step",
            y_keys=("ego_cumulative_reward", "enm_cumulative_reward"),
            labels=("Ego Cumulative Reward", "Enemy Cumulative Reward"),
            title="Cumulative Reward Over Time",
            ylabel="Reward",
            save_path=os.path.join(episode_dir, "reward_curve.png"),
        )
        self._plot_two_team_line(
            x_key="step",
            y_keys=("ego_active_missiles", "enm_active_missiles"),
            labels=("Ego Active Missiles", "Enemy Active Missiles"),
            title="Active Missiles Over Time",
            ylabel="Missile Count",
            save_path=os.path.join(episode_dir, "active_missiles_curve.png"),
        )
        self._plot_two_team_line(
            x_key="step",
            y_keys=("ego_avg_nearest_enemy_distance_m", "enm_avg_nearest_enemy_distance_m"),
            labels=("Ego Avg Nearest Distance", "Enemy Avg Nearest Distance"),
            title="Average Nearest Enemy Distance",
            ylabel="Distance (m)",
            save_path=os.path.join(episode_dir, "distance_curve.png"),
        )

    def _plot_summary_curves(self, out_dir):
        episodes = [int(row["episode"]) for row in self.episode_rows]
        winners = [row["winner"] for row in self.episode_rows]
        win_counts = {
            "ego": winners.count("ego"),
            "enm": winners.count("enm"),
            "draw": winners.count("draw"),
        }

        plt.figure(figsize=(8, 5))
        plt.bar(["ego", "enm", "draw"], [win_counts["ego"], win_counts["enm"], win_counts["draw"]])
        plt.title("Win Count Summary")
        plt.ylabel("Count")
        plt.grid(axis="y", alpha=0.3)
        plt.savefig(os.path.join(out_dir, "summary_win_count.png"), dpi=150, bbox_inches="tight")
        plt.close()

        plt.figure(figsize=(10, 6))
        plt.plot(episodes, [row["ego_reward_sum"] for row in self.episode_rows], marker="o", label="Ego Reward")
        plt.plot(episodes, [row["enm_reward_sum"] for row in self.episode_rows], marker="o", label="Enemy Reward")
        plt.title("Episode Reward Summary")
        plt.xlabel("Episode")
        plt.ylabel("Reward Sum")
        plt.legend(loc="best")
        plt.grid(alpha=0.3)
        plt.savefig(os.path.join(out_dir, "summary_reward.png"), dpi=150, bbox_inches="tight")
        plt.close()

        plt.figure(figsize=(10, 6))
        plt.plot(episodes, [row["ego_hit_rate"] for row in self.episode_rows], marker="o", label="Ego Hit Rate")
        plt.plot(episodes, [row["enm_hit_rate"] for row in self.episode_rows], marker="o", label="Enemy Hit Rate")
        plt.title("Episode Missile Hit Rate")
        plt.xlabel("Episode")
        plt.ylabel("Hit Rate")
        plt.legend(loc="best")
        plt.grid(alpha=0.3)
        plt.savefig(os.path.join(out_dir, "summary_hit_rate.png"), dpi=150, bbox_inches="tight")
        plt.close()

        plt.figure(figsize=(10, 6))
        plt.plot(episodes, [row["ego_alive_count"] for row in self.episode_rows], marker="o", label="Ego Alive")
        plt.plot(episodes, [row["enm_alive_count"] for row in self.episode_rows], marker="o", label="Enemy Alive")
        plt.title("Episode Alive Count Summary")
        plt.xlabel("Episode")
        plt.ylabel("Alive Count")
        plt.legend(loc="best")
        plt.grid(alpha=0.3)
        plt.savefig(os.path.join(out_dir, "summary_alive_count.png"), dpi=150, bbox_inches="tight")
        plt.close()

    def _plot_two_team_line(self, x_key, y_keys, labels, title, ylabel, save_path):
        plt.figure(figsize=(12, 6))
        x = [row[x_key] for row in self._current_episode_steps]
        for y_key, label in zip(y_keys, labels):
            y = [row[y_key] for row in self._current_episode_steps]
            plt.plot(x, y, linewidth=2, label=label)
        plt.title(title)
        plt.xlabel("Step")
        plt.ylabel(ylabel)
        plt.legend(loc="best")
        plt.grid(alpha=0.3)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

    def _mean_of_column(self, key):
        return self._nanmean([row.get(key, np.nan) for row in self._current_episode_steps])

    @staticmethod
    def _write_csv(csv_path, rows):
        if not rows:
            return
        fieldnames = sorted({key for row in rows for key in row.keys()})
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

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

