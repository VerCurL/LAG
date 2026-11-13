"""C10-D6: pure energy-balance command probe on the frozen C10 stack."""

import math

import numpy as np

from .baiyepower_energy_command_c10_r2_task import (
    BaiyePowerEnergyCommandC10R2Task,
)
from .baiyepower_energy_exchange_c7_r0_task import (
    BaiyePowerEnergyExchangeC7R0Task,
)
from .baiyepower_v2_task import BaiyePowerV2Task


class BaiyePowerEnergyBalanceProbeC10D6Task(
    BaiyePowerEnergyCommandC10R2Task
):
    """Expose only a bounded flight-path offset, never actuator residuals."""

    FROZEN_TASK_NAME = "baiyepower_energy_balance_probe_c10_d6"
    FROZEN_OBS_VERSION = "v5_target_rates"
    FROZEN_TARGET_RATES_ENABLED = True
    FROZEN_EXPERIMENT_ID = "C10-D6"
    FROZEN_MODE = "energy_balance_probe"
    FROZEN_SURFACE_RESIDUAL_SCALE = np.zeros(4, dtype=np.float32)
    FROZEN_COMMAND_ACTION_INDEX = 3
    FROZEN_VERTICAL_SPEED_COMMAND_SCALE_MPS = 1.0
    FROZEN_SPEED_THROTTLE_GAIN = 0.01
    FROZEN_C7_VALUES = BaiyePowerEnergyExchangeC7R0Task.FROZEN_VALUES

    def _validate_isolation_config(self):
        # Bypass the C3/C10 vertical_speed_command restriction while retaining
        # the common BaiyePower-v2 observation, QP, lifecycle and safety gates.
        BaiyePowerV2Task._validate_isolation_config(self)
        experiment_id = self.FROZEN_EXPERIMENT_ID
        if str(getattr(self.config, "task", "")) != self.FROZEN_TASK_NAME:
            raise ValueError("{} requires its isolated task name".format(
                experiment_id
            ))
        if (
            str(getattr(self.config, "obs_version", "")).strip().lower()
            != self.FROZEN_OBS_VERSION
        ):
            raise ValueError(
                "{} requires obs_version={}".format(
                    experiment_id,
                    self.FROZEN_OBS_VERSION,
                )
            )
        if not bool(
            getattr(self.config, "BaiyePowerTargetRates_enable", False)
        ):
            raise ValueError("{} requires target-rate observations".format(
                experiment_id
            ))
        if str(
            getattr(self.config, "BaiyePowerTargetRates_source", "")
        ).strip().lower() != "route_analytic_v1":
            raise ValueError(
                "{} target rates must use route_analytic_v1".format(
                    experiment_id
                )
            )
        if str(
            getattr(self.config, "BaiyePowerTargetRates_experiment_id", "")
        ).strip() != experiment_id:
            raise ValueError(
                "{} requires its target-rate experiment id".format(
                    experiment_id
                )
            )
        target_rate_scales = np.asarray(
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
        if not np.array_equal(
            target_rate_scales,
            self.FROZEN_TARGET_RATE_SCALES,
        ):
            raise ValueError(
                "{} target-rate scales must remain [3, 10, 0.2]".format(
                    experiment_id
                )
            )

        if not bool(
            getattr(self.config, "CoordinatedResidual_enable", False)
        ):
            raise ValueError(
                "{} requires CoordinatedResidual_enable=true".format(
                    experiment_id
                )
            )
        if str(
            getattr(self.config, "CoordinatedResidual_mode", "")
        ).strip().lower() != self.FROZEN_MODE:
            raise ValueError(
                "{} requires energy_balance_probe mode".format(
                    experiment_id
                )
            )
        if str(
            getattr(self.config, "CoordinatedResidual_rate_source", "")
        ).strip().lower() != "route_analytic_v1":
            raise ValueError(
                "{} controller target rates must use route_analytic_v1".format(
                    experiment_id
                )
            )
        if str(
            getattr(self.config, "CoordinatedResidual_experiment_id", "")
        ).strip() != experiment_id:
            raise ValueError(
                "{} requires its frozen experiment id".format(experiment_id)
            )

        residual_scale = np.asarray(
            getattr(
                self.config,
                "CoordinatedResidual_residual_scale",
                [],
            ),
            dtype=np.float32,
        ).reshape(-1)
        if (
            residual_scale.shape != (4,)
            or not np.array_equal(
                residual_scale,
                self.FROZEN_SURFACE_RESIDUAL_SCALE,
            )
        ):
            raise ValueError(
                "{} direct actuator residual scale must remain zero".format(
                    experiment_id
                )
            )
        action_index = getattr(
            self.config,
            "CoordinatedResidual_command_action_index",
            None,
        )
        if (
            isinstance(action_index, bool)
            or action_index != self.FROZEN_COMMAND_ACTION_INDEX
        ):
            raise ValueError(
                "{} command action index is frozen at 3".format(
                    experiment_id
                )
            )
        scalar_values = {
            "CoordinatedResidual_vertical_speed_command_scale_mps": (
                self.FROZEN_VERTICAL_SPEED_COMMAND_SCALE_MPS
            ),
            "CoordinatedResidual_speed_throttle_gain": (
                self.FROZEN_SPEED_THROTTLE_GAIN
            ),
        }
        scalar_values.update(self.FROZEN_C7_VALUES)
        for name, expected in scalar_values.items():
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
                raise ValueError("{} is frozen at {}".format(name, expected))

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
            getattr(self.config, "F16FlaperonCommon_experiment_id", "")
        ).strip() != "C9-M1":
            raise ValueError("{} requires the C9-M1 plant".format(
                experiment_id
            ))
        if not bool(getattr(self.config, "JSBSim_dt_before_load", False)):
            raise ValueError("{} requires dt-before-load".format(
                experiment_id
            ))
        aircraft = getattr(self.config, "aircraft_configs", {}).get(
            "A0100",
            {},
        )
        if aircraft.get("model") != "f16_flaperon_common_m0":
            raise ValueError(
                "{} requires f16_flaperon_common_m0".format(experiment_id)
            )
