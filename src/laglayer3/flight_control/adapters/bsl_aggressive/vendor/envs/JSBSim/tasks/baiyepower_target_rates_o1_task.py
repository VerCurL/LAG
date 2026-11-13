"""Isolated O1 experiment: U1 plus analytic target-rate observation."""

from .baiyepower_upright_v1_task import BaiyePowerUprightV1Task


class BaiyePowerTargetRatesO1Task(BaiyePowerUprightV1Task):
    """Append three target rates while freezing every U1 control objective."""

    def _validate_isolation_config(self):
        super()._validate_isolation_config()
        if str(getattr(self.config, "obs_version", "")).lower() != (
            "v5_target_rates"
        ):
            raise ValueError(
                "baiyepower_target_rates_o1 requires v5_target_rates"
            )
        if not bool(
            getattr(
                self.config,
                "BaiyePowerTargetRates_enable",
                False,
            )
        ):
            raise ValueError(
                "baiyepower_target_rates_o1 requires target rates"
            )
        if str(
            getattr(
                self.config,
                "BaiyePowerTargetRates_source",
                "",
            )
        ) != "route_analytic_v1":
            raise ValueError(
                "O1 target rates must use route_analytic_v1"
            )

    def begin_transition(self, env):
        super().begin_transition(env)
        for agent_id in env.agents:
            rates = self.get_target_rate_commands(env, agent_id)
            self.record_baiyepower_v2_diag(
                agent_id,
                {
                    "target_heading_rate_deg_s": rates[0],
                    "target_altitude_rate_mps": rates[1],
                    "target_speed_rate_mps2": rates[2],
                    "target_rates_observation_active": 1.0,
                },
            )
