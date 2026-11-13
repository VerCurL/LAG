"""C10-S0: conservative command-residual smoke on the C9-M1 plant."""

import math

import numpy as np

from .baiyepower_coordinated_command_c3_v1_task import (
    BaiyePowerCoordinatedCommandC3V1Task,
)
from .baiyepower_energy_exchange_c7_r0_task import (
    BaiyePowerEnergyExchangeC7R0Task,
)


class BaiyePowerEnergyCommandC10S0Task(
    BaiyePowerCoordinatedCommandC3V1Task
):
    """Let PPO adjust one slow command around the frozen C7-R0 nominal law."""

    FROZEN_TASK_NAME = "baiyepower_energy_command_c10_s0"
    FROZEN_EXPERIMENT_ID = "C10-S0"
    FROZEN_VERTICAL_SPEED_COMMAND_SCALE_MPS = 1.0
    FROZEN_C7_VALUES = BaiyePowerEnergyExchangeC7R0Task.FROZEN_VALUES

    def _validate_isolation_config(self):
        super()._validate_isolation_config()
        experiment_id = self.FROZEN_EXPERIMENT_ID
        if str(
            getattr(
                self.config,
                "CoordinatedResidual_energy_allocator_mode",
                "",
            )
        ).strip().lower() != "adaptive_total_energy_throttle_v1":
            raise ValueError(
                "{} must retain adaptive_total_energy_throttle_v1".format(
                    experiment_id
                )
            )
        if str(
            getattr(
                self.config,
                "CoordinatedResidual_energy_exchange_mode",
                "",
            )
        ).strip().lower() != "measured_opposite_sign_v1":
            raise ValueError(
                "{} must retain measured_opposite_sign_v1".format(
                    experiment_id
                )
            )
        if str(
            getattr(
                self.config,
                "F16FlaperonCommon_experiment_id",
                "",
            )
        ).strip() != "C9-M1":
            raise ValueError(
                "{} requires the C9-M1 physical model".format(
                    experiment_id
                )
            )
        if not bool(
            getattr(self.config, "JSBSim_dt_before_load", False)
        ):
            raise ValueError(
                "{} requires dt-before-load".format(experiment_id)
            )

        aircraft_configs = getattr(self.config, "aircraft_configs", {})
        aircraft = aircraft_configs.get("A0100", {})
        if aircraft.get("model") != "f16_flaperon_common_m0":
            raise ValueError(
                "{} requires f16_flaperon_common_m0".format(
                    experiment_id
                )
            )

        for name, expected in self.FROZEN_C7_VALUES.items():
            actual = float(getattr(self.config, name, np.nan))
            if (
                not math.isfinite(actual)
                or not math.isclose(
                    actual,
                    expected,
                    rel_tol=0.0,
                    abs_tol=1e-15,
                )
            ):
                raise ValueError(
                    "{} is frozen at {}".format(name, expected)
                )
