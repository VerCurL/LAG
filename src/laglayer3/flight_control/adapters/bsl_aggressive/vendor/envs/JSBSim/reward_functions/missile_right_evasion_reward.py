import numpy as np

from .reward_function_base import BaseRewardFunction
from ..core.catalog import Catalog as c


class MissileRightEvasionReward(BaseRewardFunction):
    """
    Encourage a natural right-evade behavior shortly after missile warning appears.
    This is reward shaping, not hard action override.
    """

    def __init__(self, config):
        super().__init__(config)
        cls = self.__class__.__name__

        self.enable = getattr(config, f"{cls}_enable", False)
        self.alert_steps = int(getattr(config, f"{cls}_alert_steps", 8))
        self.alert_only = getattr(config, f"{cls}_alert_only", True)

        self.aileron_threshold = float(getattr(config, f"{cls}_aileron_threshold", 0.25))
        self.rudder_threshold = float(getattr(config, f"{cls}_rudder_threshold", 0.10))
        self.target_roll_rad = float(getattr(config, f"{cls}_target_roll_rad", 0.70))

        self.cmd_weight = float(getattr(config, f"{cls}_cmd_weight", 0.7))
        self.roll_weight = float(getattr(config, f"{cls}_roll_weight", 0.3))
        # Decision gate (from tactical pipeline): encourage stronger evade
        # when permit/threat are high and suppress in calm.
        self.use_decision_gate = getattr(config, f"{cls}_use_decision_gate", True)
        self.permit_gain = float(getattr(config, f"{cls}_permit_gain", 1.0))
        self.threat_gain = float(getattr(config, f"{cls}_threat_gain", 0.8))
        self.calm_suppression = float(getattr(config, f"{cls}_calm_suppression", 0.6))
        self.min_multiplier = float(getattr(config, f"{cls}_min_multiplier", 0.25))
        self.max_multiplier = float(getattr(config, f"{cls}_max_multiplier", 3.5))

        self._alert_countdown = {}
        self._prev_warning = {}

        self.reward_item_names = [
            self.__class__.__name__,
            self.__class__.__name__ + "_cmd",
            self.__class__.__name__ + "_roll",
            self.__class__.__name__ + "_alert",
        ]

    def reset(self, task, env):
        self._alert_countdown = {agent_id: 0 for agent_id in env.agents.keys()}
        self._prev_warning = {agent_id: False for agent_id in env.agents.keys()}
        return super().reset(task, env)

    def _right_cmd_score(self, aileron_cmd, rudder_cmd):
        a_score = np.clip(
            (aileron_cmd - self.aileron_threshold) / max(1.0 - self.aileron_threshold, 1e-6),
            0.0,
            1.0,
        )
        r_score = np.clip(
            (rudder_cmd - self.rudder_threshold) / max(1.0 - self.rudder_threshold, 1e-6),
            0.0,
            1.0,
        )
        return 0.7 * a_score + 0.3 * r_score

    def _roll_score(self, roll_rad):
        return np.clip(roll_rad / max(abs(self.target_roll_rad), 1e-6), -1.0, 1.0)

    def _decision_multiplier(self, task, agent_id):
        if not self.use_decision_gate or not hasattr(task, "get_decision_ctx"):
            return 1.0
        ctx = task.get_decision_ctx(agent_id) or {}
        permit = float(np.clip(ctx.get("permit", 0.0), 0.0, 1.0))
        s_threat = float(np.clip(ctx.get("s_threat", 0.0), 0.0, 1.0))

        boost = self.permit_gain * permit + self.threat_gain * s_threat
        suppress = self.calm_suppression * (1.0 - permit) * (1.0 - s_threat)
        multiplier = 1.0 + boost - suppress
        return float(np.clip(multiplier, self.min_multiplier, self.max_multiplier))

    def get_reward(self, task, env, agent_id):
        if not self.enable:
            return self._process(0.0, agent_id, (0.0, 0.0, 0.0))

        sim = env.agents[agent_id]
        warning_now = sim.check_missile_warning() is not None
        warning_prev = self._prev_warning.get(agent_id, False)

        # Rising edge: newly detected missile warning
        if warning_now and not warning_prev:
            self._alert_countdown[agent_id] = self.alert_steps
        self._prev_warning[agent_id] = warning_now

        countdown = self._alert_countdown.get(agent_id, 0)
        alert_active = warning_now or countdown > 0
        if countdown > 0:
            self._alert_countdown[agent_id] = countdown - 1

        if self.alert_only and not alert_active:
            return self._process(0.0, agent_id, (0.0, 0.0, 0.0))

        aileron_cmd = sim.get_property_value(c.fcs_aileron_cmd_norm)
        rudder_cmd = sim.get_property_value(c.fcs_rudder_cmd_norm)
        roll_rad = sim.get_property_value(c.attitude_roll_rad)

        cmd_score = self._right_cmd_score(aileron_cmd, rudder_cmd)
        roll_score = self._roll_score(roll_rad)

        reward = self.cmd_weight * cmd_score + self.roll_weight * roll_score
        reward *= self._decision_multiplier(task, agent_id)
        return self._process(reward, agent_id, (cmd_score, roll_score, float(alert_active)))
