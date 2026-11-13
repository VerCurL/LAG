"""C10-D10: route reward by the frozen slow action's control authority."""

import math

from .baiyepower_energy_balance_probe_c10_d6_task import (
    BaiyePowerEnergyBalanceProbeC10D6Task,
)


class BaiyePowerAuthorityRoutedC10D10Task(
    BaiyePowerEnergyBalanceProbeC10D6Task
):
    """Keep direction diagnostics, but route them out of slow-action reward."""

    FROZEN_TASK_NAME = "baiyepower_authority_routed_c10_d10"
    FROZEN_EXPERIMENT_ID = "C10-D10"
    FROZEN_REWARD_ROUTING_MODE = "energy_safety_for_slow_command_v1"

    def _validate_isolation_config(self):
        super()._validate_isolation_config()
        mode = str(
            getattr(self.config, "BaiyePowerRewardRouting_mode", "")
        ).strip().lower()
        if mode != self.FROZEN_REWARD_ROUTING_MODE:
            raise ValueError(
                "C10-D10 reward routing mode must remain {}".format(
                    self.FROZEN_REWARD_ROUTING_MODE
                )
            )
        experiment_id = str(
            getattr(
                self.config,
                "BaiyePowerRewardRouting_experiment_id",
                "",
            )
        ).strip()
        if experiment_id != self.FROZEN_EXPERIMENT_ID:
            raise ValueError(
                "C10-D10 requires its reward-routing experiment id"
            )

    def get_reward(self, env, agent_id, info=None):
        aggregate_reward, info = super().get_reward(
            env, agent_id, info
        )
        diag = self._transition_diag.setdefault(agent_id, {})
        direction_reward = float(
            diag.get("direction_reward_contribution", 0.0)
        )
        if not math.isfinite(direction_reward):
            raise RuntimeError(
                "C10-D10 direction reward diagnostic is non-finite"
            )
        routed_reward = float(aggregate_reward) - direction_reward
        if not math.isfinite(routed_reward):
            raise RuntimeError("C10-D10 routed reward is non-finite")
        diag["direction_reward_diagnostic_only"] = direction_reward
        diag["direction_reward_routed_out"] = direction_reward
        diag["reward_contribution_sum_before_routing"] = float(
            aggregate_reward
        )
        diag["reward_contribution_sum"] = routed_reward
        diag["reward_routing_active"] = 1.0
        return routed_reward, info
