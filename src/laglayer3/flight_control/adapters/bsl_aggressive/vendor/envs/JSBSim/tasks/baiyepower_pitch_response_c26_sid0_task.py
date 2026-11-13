"""C26-SID0: isolated maneuver-only elevator response identification."""

import math

import numpy as np

from .baiyepower_recentered_residual_c17_r0_task import (
    BaiyePowerRecenteredResidualC17R0Task,
)
from .baiyepower_v2_task import BaiyePowerV2Task
from ..utils.utils import parse_config


class BaiyePowerPitchResponseC26SID0Task(
    BaiyePowerRecenteredResidualC17R0Task
):
    """Retain the C17/C26 stack while opening only a 0.05 elevator probe."""

    FROZEN_TASK_NAME = "baiyepower_pitch_response_c26_sid0"
    FROZEN_EXPERIMENT_ID = "C26-SID0"
    FROZEN_MODE = "residual"
    FROZEN_SURFACE_RESIDUAL_SCALE = np.asarray(
        [0.0, 0.05, 0.0, 0.0], dtype=np.float32
    )
    ALLOWED_CONFIG_DIFFERENCES = {
        "task",
        "CoordinatedResidual_experiment_id",
        "BaiyePowerTargetRates_experiment_id",
        "BaiyePowerRewardRouting_experiment_id",
        "DirectionInvariantObservation_experiment_id",
        "StructuredResidual_experiment_id",
        "CoordinatedResidual_mode",
        "CoordinatedResidual_residual_scale",
    }

    def _validate_isolation_config(self):
        # The inherited C17 validator deliberately prohibits actuator
        # residuals.  SID0 bypasses that single experimental restriction, then
        # proves by full resolved-config comparison that no unrelated setting
        # changed from the N5 operating point.
        BaiyePowerV2Task._validate_isolation_config(self)
        reference = parse_config(
            "1/baiyepower_pitch_factorial_c26_k075_g025"
        )
        current_values = {
            key: value
            for key, value in vars(self.config).items()
            if not key.startswith("_")
        }
        reference_values = {
            key: value
            for key, value in vars(reference).items()
            if not key.startswith("_")
        }
        for key in sorted(set(current_values) | set(reference_values)):
            if key in self.ALLOWED_CONFIG_DIFFERENCES:
                continue
            if key not in current_values or key not in reference_values:
                raise ValueError(
                    "C26-SID0 unexpected config field difference: {}".format(
                        key
                    )
                )
            actual = current_values[key]
            expected = reference_values[key]
            if isinstance(actual, (list, tuple)) or isinstance(
                expected, (list, tuple)
            ):
                matches = np.array_equal(
                    np.asarray(actual), np.asarray(expected)
                )
            else:
                matches = actual == expected
            if not matches:
                raise ValueError(
                    "C26-SID0 changed frozen field {}".format(key)
                )
        if str(getattr(self.config, "task", "")) != self.FROZEN_TASK_NAME:
            raise ValueError("C26-SID0 requires its isolated task identity")
        for field in (
            "CoordinatedResidual_experiment_id",
            "BaiyePowerTargetRates_experiment_id",
            "BaiyePowerRewardRouting_experiment_id",
            "DirectionInvariantObservation_experiment_id",
            "StructuredResidual_experiment_id",
        ):
            if str(getattr(self.config, field, "")) != self.FROZEN_EXPERIMENT_ID:
                raise ValueError("C26-SID0 identity mismatch: {}".format(field))
        if str(
            getattr(self.config, "CoordinatedResidual_mode", "")
        ).strip().lower() != self.FROZEN_MODE:
            raise ValueError("C26-SID0 requires residual mode")
        residual_scale = np.asarray(
            getattr(self.config, "CoordinatedResidual_residual_scale", ()),
            dtype=np.float32,
        ).reshape(-1)
        if not np.array_equal(residual_scale, self.FROZEN_SURFACE_RESIDUAL_SCALE):
            raise ValueError("C26-SID0 opens only 0.05 elevator residual")
        if not math.isclose(
            float(
                getattr(
                    self.config,
                    "CoordinatedResidual_turn_pitch_coupling_gain",
                    math.nan,
                )
            ),
            0.75,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("C26-SID0 requires turn-pitch coupling 0.75")
        if not math.isclose(
            float(
                getattr(
                    self.config,
                    "CoordinatedResidual_pitch_rate_feedback_gain",
                    math.nan,
                )
            ),
            0.25,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("C26-SID0 requires pitch-rate feedback 0.25")
