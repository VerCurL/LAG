import csv
import os
from collections import defaultdict
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np


class FlightDataRecorder:
    """
    Record per-step flight metrics and reward breakdown for env 0 only.
    """

    def __init__(
        self,
        save_dir: str,
        csv_name_prefix: str = "flight_trace",
        tracked_agent_id: Optional[str] = None,
        plot_agent_ids: Optional[List[str]] = None,
    ):
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.csv_name_prefix = csv_name_prefix
        self.tracked_agent_id = tracked_agent_id
        self.plot_agent_ids = set(plot_agent_ids) if plot_agent_ids else None
        self.rows: List[Dict[str, float]] = []
        self.last_dumped_episode: Optional[int] = None

    def record_infos(self, infos, episode: int, rollout_step: int, total_num_steps: int):
        """
        Parse vec-env infos and record one row per agent for env 0 only.
        """
        if infos is None or len(infos) == 0:
            return

        info = infos[0]
        if not isinstance(info, dict):
            return

        reward_breakdown = info.get("reward_breakdown", {})
        flight_metrics = info.get("flight_metrics", {})
        current_step = info.get("current_step", rollout_step + 1)

        agent_ids = sorted(set(reward_breakdown.keys()) | set(flight_metrics.keys()))
        for agent_id in agent_ids:
            if self.tracked_agent_id and agent_id != self.tracked_agent_id:
                continue

            row = {
                "episode": int(episode),
                "rollout_step": int(rollout_step),
                "env_step": int(current_step),
                "total_num_steps": int(total_num_steps),
                "env_index": 0,
                "agent_id": agent_id,
            }

            metrics = flight_metrics.get(agent_id, {})
            for key, value in metrics.items():
                row[key] = self._to_float(value)

            rewards = reward_breakdown.get(agent_id, {})
            for key, value in rewards.items():
                row[f"reward_{key}"] = self._to_float(value)

            self.rows.append(row)

    def dump_episode(self, episode: int, draw_plot: bool = False):
        """
        Dump current episode data for env 0 into:
          save_dir/<episode>/flight_trace_env0.csv
        """
        if not self.rows:
            return

        episode_dir = os.path.join(self.save_dir, str(int(episode)))
        os.makedirs(episode_dir, exist_ok=True)

        csv_path = os.path.join(episode_dir, f"{self.csv_name_prefix}_env0.csv")
        self._write_rows_to_csv(self.rows, csv_path)
        if draw_plot:
            self._plot_rows(self.rows, episode_dir, 0)

        self.rows.clear()
        self.last_dumped_episode = int(episode)

    def close(self):
        """
        Keep close lightweight; episode-level dumping is controlled by runner.
        """
        return

    def _plot_rows(self, rows: List[Dict[str, float]], out_dir: str, env_index: int):
        if not rows:
            return

        env_dir = os.path.join(out_dir, str(int(env_index)))
        os.makedirs(env_dir, exist_ok=True)

        grouped_by_agent = defaultdict(list)
        for row in rows:
            grouped_by_agent[str(row["agent_id"])].append(row)

        if self.plot_agent_ids is not None:
            grouped_by_agent = {
                agent_id: agent_rows
                for agent_id, agent_rows in grouped_by_agent.items()
                if agent_id in self.plot_agent_ids
            }
            if not grouped_by_agent:
                return

        for agent_id in grouped_by_agent.keys():
            grouped_by_agent[agent_id] = sorted(
                grouped_by_agent[agent_id],
                key=lambda x: (x.get("episode", 0), x.get("rollout_step", 0)),
            )

        meta_fields = {"episode", "rollout_step", "env_step", "total_num_steps", "env_index", "agent_id"}
        metric_keys = sorted(
            [
                key
                for key in {k for row in rows for k in row.keys()}
                if key not in meta_fields and not key.startswith("reward_")
            ]
        )
        reward_keys = sorted(
            [
                key
                for key in {k for row in rows for k in row.keys()}
                if key.startswith("reward_")
            ]
        )

        # Category 1: one metric per figure, compare all agents in the same figure.
        for metric_key in metric_keys:
            plt.figure(figsize=(14, 6))
            for agent_id, agent_rows in grouped_by_agent.items():
                x = [self._to_float(row.get("rollout_step", idx)) for idx, row in enumerate(agent_rows)]
                y = [self._to_float(row.get(metric_key, np.nan)) for row in agent_rows]
                plt.plot(x, y, label=agent_id)
            plt.title(f"Env {env_index} - {metric_key}")
            plt.xlabel("Rollout Step")
            plt.ylabel(metric_key)
            plt.legend(loc="best")
            plt.grid(alpha=0.3)
            metric_path = os.path.join(env_dir, f"metric_{metric_key}.png")
            plt.savefig(metric_path, dpi=150)
            plt.close()

        # Category 2: one figure per reward, compare all selected agents in the same figure.
        for reward_key in reward_keys:
            plt.figure(figsize=(14, 6))
            has_any_curve = False
            for agent_id, agent_rows in grouped_by_agent.items():
                x = [self._to_float(row.get("rollout_step", idx)) for idx, row in enumerate(agent_rows)]
                y = [self._to_float(row.get(reward_key, np.nan)) for row in agent_rows]
                if np.all(np.isnan(y)):
                    continue
                has_any_curve = True
                plt.plot(x, y, label=agent_id)

            if not has_any_curve:
                plt.close()
                continue

            reward_name = reward_key.replace("reward_", "")
            plt.title(f"Env {env_index} - Reward {reward_name}")
            plt.xlabel("Rollout Step")
            plt.ylabel("Reward Value")
            plt.legend(loc="best")
            plt.grid(alpha=0.3)
            reward_path = os.path.join(env_dir, f"reward_{reward_name}.png")
            plt.savefig(reward_path, dpi=150)
            plt.close()

    @staticmethod
    def _write_rows_to_csv(rows: List[Dict[str, float]], csv_path: str):
        fieldnames = sorted({key for row in rows for key in row.keys()})
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def plot_csv(self, csv_path: str, out_dir: Optional[str] = None):
        if not os.path.exists(csv_path):
            return
        rows = []
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                parsed = {}
                for key, value in row.items():
                    if key == "agent_id":
                        parsed[key] = value
                    else:
                        parsed[key] = self._to_float(value)
                rows.append(parsed)
        if not rows:
            return
        out_dir = out_dir or os.path.dirname(csv_path)
        env_index = int(rows[0].get("env_index", 0))
        self._plot_rows(rows, out_dir, env_index)

    @staticmethod
    def _to_float(value):
        try:
            return float(value)
        except (TypeError, ValueError):
            return np.nan
