"""C3-V2 task: C3 command residual plus explicit target-rate observation."""

import numpy as np

from .baiyepower_coordinated_command_c3_v1_task import (
    BaiyePowerCoordinatedCommandC3V1Task,
)


class BaiyePowerCoordinatedCommandC3V2Task(
    BaiyePowerCoordinatedCommandC3V1Task
):
    """Expose route derivatives without changing C3-V1 control authority."""

    FROZEN_TASK_NAME = "baiyepower_coordinated_command_c3_v2"
    FROZEN_OBS_VERSION = "v5_target_rates"
    FROZEN_TARGET_RATES_ENABLED = True
    FROZEN_EXPERIMENT_ID = "C3-V2"
    FROZEN_TARGET_RATE_SCALES = np.asarray(
        [3.0, 10.0, 0.2],
        dtype=np.float32,
    )

    def _validate_isolation_config(self):
        super()._validate_isolation_config()
        if str(
            getattr(
                self.config,
                "BaiyePowerTargetRates_source",
                "",
            )
        ).strip().lower() != "route_analytic_v1":
            raise ValueError(
                "C3-V2 target rates must use route_analytic_v1"
            )
        if str(
            getattr(
                self.config,
                "BaiyePowerTargetRates_experiment_id",
                "",
            )
        ).strip() != "C3-V2":
            raise ValueError(
                "C3-V2 requires target-rate experiment id C3-V2"
            )

        scales = np.asarray(
            [
                getattr(
                    self.config,
                    "TargetRateObservation_heading_scale_deg_s",
                    np.nan,
                ),
                getattr(
                    self.config,
                    "TargetRateObservation_altitude_scale_mps",
                    np.nan,
                ),
                getattr(
                    self.config,
                    "TargetRateObservation_speed_scale_mps2",
                    np.nan,
                ),
            ],
            dtype=np.float32,
        )
        if (
            not np.all(np.isfinite(scales))
            or not np.array_equal(
                scales,
                self.FROZEN_TARGET_RATE_SCALES,
            )
        ):
            raise ValueError(
                "C3-V2 target-rate scales are frozen at "
                "[3.0 deg/s, 10.0 m/s, 0.2 m/s2]"
            )

    def begin_transition(self, env):
        super().begin_transition(env)
        for agent_id in env.agents:
            rates = self.get_target_rate_commands(env, agent_id)
            if (
                rates.shape != (3,)
                or not np.all(np.isfinite(rates))
            ):
                raise RuntimeError(
                    "C3-V2 target-rate observation must be finite"
                )
            self.record_baiyepower_v2_diag(
                agent_id,
                {
                    "target_heading_rate_deg_s": rates[0],
                    "target_altitude_rate_mps": rates[1],
                    "target_speed_rate_mps2": rates[2],
                    "target_rates_observation_active": 1.0,
                },
            )
