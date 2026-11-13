"""C3-V1 task: learned vertical-speed command residual."""

import math

import numpy as np

from .baiyepower_upright_v1_task import BaiyePowerUprightV1Task


class BaiyePowerCoordinatedCommandC3V1Task(BaiyePowerUprightV1Task):
    """Let PPO change one slow reference, never a direct actuator."""

    FROZEN_TASK_NAME = "baiyepower_coordinated_command_c3_v1"
    FROZEN_OBS_VERSION = "v4_qp_state"
    FROZEN_TARGET_RATES_ENABLED = False
    FROZEN_EXPERIMENT_ID = "C3-V1"
    FROZEN_SURFACE_RESIDUAL_SCALE = np.zeros(4, dtype=np.float32)
    FROZEN_COMMAND_ACTION_INDEX = 3
    FROZEN_VERTICAL_SPEED_COMMAND_SCALE_MPS = 5.0
    FROZEN_SPEED_THROTTLE_GAIN = 0.01

    def _validate_isolation_config(self):
        super()._validate_isolation_config()
        experiment_id = self.FROZEN_EXPERIMENT_ID
        if str(getattr(self.config, "task", "")) != self.FROZEN_TASK_NAME:
            raise ValueError(
                "{} requires its isolated task name".format(
                    experiment_id
                )
            )
        if (
            str(getattr(self.config, "obs_version", "")).lower()
            != self.FROZEN_OBS_VERSION
        ):
            raise ValueError(
                "{} requires obs_version={}".format(
                    experiment_id,
                    self.FROZEN_OBS_VERSION,
                )
            )
        target_rates_enabled = bool(
            getattr(
                self.config,
                "BaiyePowerTargetRates_enable",
                False,
            )
        )
        if target_rates_enabled != self.FROZEN_TARGET_RATES_ENABLED:
            if self.FROZEN_TARGET_RATES_ENABLED:
                raise ValueError(
                    "{} requires target-rate observations".format(
                        experiment_id
                    )
                )
            else:
                raise ValueError(
                    "{} cannot enable target-rate observations".format(
                        experiment_id
                    )
                )
        if not bool(
            getattr(self.config, "CoordinatedResidual_enable", False)
        ):
            raise ValueError(
                "baiyepower_coordinated_command_c3_v1 requires "
                "CoordinatedResidual_enable=true"
            )
        mode = str(
            getattr(self.config, "CoordinatedResidual_mode", "")
        ).strip().lower()
        if mode != "vertical_speed_command":
            raise ValueError(
                "{} requires vertical_speed_command mode".format(
                    experiment_id
                )
            )
        if str(
            getattr(self.config, "CoordinatedResidual_rate_source", "")
        ).strip().lower() != "route_analytic_v1":
            raise ValueError(
                "{} controller target rates must use "
                "route_analytic_v1".format(experiment_id)
            )
        if str(
            getattr(self.config, "CoordinatedResidual_experiment_id", "")
        ).strip() != self.FROZEN_EXPERIMENT_ID:
            raise ValueError(
                "{} requires its frozen experiment id".format(
                    experiment_id
                )
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
            or not np.all(np.isfinite(residual_scale))
            or not np.array_equal(
                residual_scale,
                self.FROZEN_SURFACE_RESIDUAL_SCALE,
            )
        ):
            raise ValueError(
                "{} direct actuator residual scale is frozen at "
                "[0.0, 0.0, 0.0, 0.0]".format(experiment_id)
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

        command_scale_mps = float(
            getattr(
                self.config,
                "CoordinatedResidual_vertical_speed_command_scale_mps",
                np.nan,
            )
        )
        if (
            not math.isfinite(command_scale_mps)
            or not math.isclose(
                command_scale_mps,
                self.FROZEN_VERTICAL_SPEED_COMMAND_SCALE_MPS,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            raise ValueError(
                "{} vertical-speed command scale is frozen at "
                "{:.1f} m/s".format(
                    experiment_id,
                    self.FROZEN_VERTICAL_SPEED_COMMAND_SCALE_MPS,
                )
            )

        speed_throttle_gain = float(
            getattr(
                self.config,
                "CoordinatedResidual_speed_throttle_gain",
                np.nan,
            )
        )
        if (
            not math.isfinite(speed_throttle_gain)
            or not math.isclose(
                speed_throttle_gain,
                self.FROZEN_SPEED_THROTTLE_GAIN,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            raise ValueError(
                "{} speed-to-throttle gain is frozen at "
                "C1's 0.01".format(experiment_id)
            )
