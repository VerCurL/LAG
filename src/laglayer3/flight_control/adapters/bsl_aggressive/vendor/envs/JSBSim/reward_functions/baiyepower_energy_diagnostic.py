from .reward_function_base import BaseRewardFunction


class BaiyePowerEnergyDiagnostic(BaseRewardFunction):
    """Zero-reward sidecar for the frozen baiyepower energy definition.

    This class deliberately contributes exactly zero to the PPO return.  Its
    trajectory exposes the signed target-minus-current decomposition needed
    to choose a reward kernel in a later, separately controlled experiment.
    """

    def __init__(self, config):
        super().__init__(config)
        self.enable = bool(
            getattr(config, "BaiyePowerEnergyDiagnostic_enable", True)
        )
        self.reward_item_names = [
            "BaiyePowerEnergyDiagnostic",
            "BaiyePower_speed_mps",
            "BaiyePower_target_speed_mps",
            "BaiyePower_energy_height_m",
            "BaiyePower_target_energy_height_m",
            "BaiyePower_altitude_deficit_m",
            "BaiyePower_kinetic_height_deficit_m",
            "BaiyePower_total_energy_height_deficit_m",
            "BaiyePower_vt_audit_mps",
            "BaiyePower_speed_minus_vt_mps",
        ]

    def get_reward(self, task, env, agent_id):
        if not self.enable:
            return self._process(
                0.0,
                agent_id,
                (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            )

        terms = task.get_baiyepower_energy_terms(env.agents[agent_id])
        vt_audit_mps = task.get_vt_audit_mps(env.agents[agent_id])
        return self._process(
            0.0,
            agent_id,
            (
                terms.speed_mps,
                terms.target_speed_mps,
                terms.energy_height_m,
                terms.target_energy_height_m,
                terms.altitude_deficit_m,
                terms.kinetic_height_deficit_m,
                terms.total_energy_height_deficit_m,
                vt_audit_mps,
                terms.speed_mps - vt_audit_mps,
            ),
        )
