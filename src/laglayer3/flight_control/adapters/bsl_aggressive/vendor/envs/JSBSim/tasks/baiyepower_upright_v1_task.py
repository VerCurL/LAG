"""Isolated U1 extension of the frozen BaiyePower v2 Full task."""

from .baiyepower_v2_task import BaiyePowerV2Task
from ..reward_functions import BaiyePowerUprightReward


class BaiyePowerUprightV1Task(BaiyePowerV2Task):
    """BaiyePower v2 Full plus a gate-only upright-envelope experiment.

    The parent task remains the reproducible U0 baseline with exactly three
    rewards.  U1 adds a zero-contribution upright diagnostic reward and enables
    the direction reward's positive-yaw reliability gate.  Observation, QP,
    minimum-jerk, curriculum, energy/course/safety costs, and termination
    conditions remain inherited unchanged.
    """

    def __init__(self, config):
        super().__init__(config)
        if not bool(getattr(config, "BaiyePowerUpright_enable", False)):
            raise ValueError(
                "baiyepower_upright_v1 requires "
                "BaiyePowerUpright_enable=true"
            )
        if not bool(
            getattr(config, "BaiyePowerUpright_yaw_gate_enable", False)
        ):
            raise ValueError(
                "baiyepower_upright_v1 requires the yaw gate"
            )
        if float(
            getattr(config, "BaiyePowerUpright_reward_rate_per_sec", 0.0)
        ) != 0.0:
            raise ValueError(
                "baiyepower_upright_v1 is gate-only; additive rate must be 0"
            )
        self.reward_functions.append(BaiyePowerUprightReward(config))
