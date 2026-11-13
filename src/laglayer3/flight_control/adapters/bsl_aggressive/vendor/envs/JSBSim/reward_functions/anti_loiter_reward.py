import numpy as np

from .reward_function_base import BaseRewardFunction


class AntiLoiterReward(BaseRewardFunction):
    """
    Penalize loiter-like spinning behavior:
      - high heading/yaw rate
      - low translational progress speed

    This targets reward hacking where the policy spins in place during missile
    warning instead of producing real escape geometry progress.
    """

    def __init__(self, config):
        super().__init__(config)
        cls = self.__class__.__name__
        self.enable = getattr(config, f"{cls}_enable", False)
        self.alert_only = getattr(config, f"{cls}_alert_only", True)
        self.use_decision_gate = getattr(config, f"{cls}_use_decision_gate", True)

        self.yaw_rate_thr_deg_s = float(getattr(config, f"{cls}_yaw_rate_thr_deg_s", 35.0))
        self.progress_thr_mps = float(getattr(config, f"{cls}_progress_thr_mps", 120.0))
        self.max_yaw_term = float(getattr(config, f"{cls}_max_yaw_term", 2.0))

        self.permit_gain = float(getattr(config, f"{cls}_permit_gain", 0.8))
        self.threat_gain = float(getattr(config, f"{cls}_threat_gain", 0.8))

        self._prev_heading = {}
        self._prev_pos = {}
        self.reward_item_names = [
            self.__class__.__name__,
            self.__class__.__name__ + "_yaw_rate_deg_s",
            self.__class__.__name__ + "_progress_mps",
            self.__class__.__name__ + "_penalty_base",
            self.__class__.__name__ + "_active",
        ]

    @staticmethod
    def _wrap_to_pi(angle: float) -> float:
        return float((angle + np.pi) % (2.0 * np.pi) - np.pi)

    def _decision_multiplier(self, task, agent_id):
        if not self.use_decision_gate or not hasattr(task, "get_decision_ctx"):
            return 1.0
        ctx = task.get_decision_ctx(agent_id) or {}
        permit = float(np.clip(ctx.get("permit", 0.0), 0.0, 1.0))
        s_threat = float(np.clip(ctx.get("s_threat", 0.0), 0.0, 1.0))
        return float(1.0 + self.permit_gain * permit + self.threat_gain * s_threat)

    def reset(self, task, env):
        self._prev_heading = {
            agent_id: float(sim.get_rpy()[2])
            for agent_id, sim in env.agents.items()
        }
        self._prev_pos = {
            agent_id: np.array(sim.get_position(), dtype=np.float64)
            for agent_id, sim in env.agents.items()
        }
        return super().reset(task, env)

    def get_reward(self, task, env, agent_id):
        if not self.enable:
            return self._process(0.0, agent_id, (0.0, 0.0, 0.0, 0.0))

        sim = env.agents[agent_id]
        warning_now = sim.check_missile_warning() is not None
        if self.alert_only and (not warning_now):
            # Keep internal state updated even when inactive.
            self._prev_heading[agent_id] = float(sim.get_rpy()[2])
            self._prev_pos[agent_id] = np.array(sim.get_position(), dtype=np.float64)
            return self._process(0.0, agent_id, (0.0, 0.0, 0.0, 0.0))

        dt = float(max(env.time_interval, 1e-6))
        heading = float(sim.get_rpy()[2])
        pos = np.array(sim.get_position(), dtype=np.float64)

        prev_heading = self._prev_heading.get(agent_id, heading)
        prev_pos = self._prev_pos.get(agent_id, pos)

        yaw_rate_deg_s = abs(np.rad2deg(self._wrap_to_pi(heading - prev_heading))) / dt
        progress_mps = float(np.linalg.norm(pos - prev_pos) / dt)

        yaw_term = np.clip(
            (yaw_rate_deg_s - self.yaw_rate_thr_deg_s) / max(self.yaw_rate_thr_deg_s, 1e-6),
            0.0,
            self.max_yaw_term,
        )
        progress_term = np.clip(
            (self.progress_thr_mps - progress_mps) / max(self.progress_thr_mps, 1e-6),
            0.0,
            1.0,
        )

        penalty_base = float(yaw_term * progress_term)
        penalty = penalty_base * self._decision_multiplier(task, agent_id)
        reward = -penalty

        self._prev_heading[agent_id] = heading
        self._prev_pos[agent_id] = pos
        return self._process(reward, agent_id, (yaw_rate_deg_s, progress_mps, penalty_base, 1.0))
