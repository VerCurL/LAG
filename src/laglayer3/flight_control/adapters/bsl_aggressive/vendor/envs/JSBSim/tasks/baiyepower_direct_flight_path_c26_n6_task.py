"""C26-N6 isolated bounded direct flight-path feedback prototype."""

import math

import numpy as np

from .baiyepower_recentered_residual_c17_r0_task import (
    BaiyePowerRecenteredResidualC17R0Task,
)
from .baiyepower_v2_task import BaiyePowerV2Task
from ..utils.utils import parse_config


class BaiyePowerDirectFlightPathC26N6Task(
    BaiyePowerRecenteredResidualC17R0Task
):
    """Freeze N5 and vary only a bounded direct flight-path correction."""

    FROZEN_TASK_NAME = "baiyepower_direct_flight_path_c26_n6"
    FROZEN_EXPERIMENT_ID = "C26-N6"
    ALLOWED_GAIN_LIMIT_PAIRS = {
        (0.0, 0.0),
        (1.5, 0.05),
        (1.5, 0.10),
        (3.0, 0.05),
        (3.0, 0.10),
    }
    ALLOWED_CONFIG_DIFFERENCES = {
        "task",
        "CoordinatedResidual_experiment_id",
        "BaiyePowerTargetRates_experiment_id",
        "BaiyePowerRewardRouting_experiment_id",
        "DirectionInvariantObservation_experiment_id",
        "StructuredResidual_experiment_id",
        "CoordinatedResidual_direct_flight_path_feedback_gain_per_rad",
        "CoordinatedResidual_direct_flight_path_feedback_max_action",
        "CoordinatedResidual_flight_path_geometry",
    }

    def _validate_isolation_config(self):
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
                    "C26-N6 unexpected config field difference: {}".format(key)
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
                raise ValueError("C26-N6 changed frozen field {}".format(key))

        if str(getattr(self.config, "task", "")) != self.FROZEN_TASK_NAME:
            raise ValueError("C26-N6 requires its isolated task identity")
        if getattr(self.config, "CoordinatedResidual_flight_path_geometry", None) != "atan2_horizontal":
            raise ValueError("C26-N6 revision 2 requires atan2_horizontal geometry")
        for field in (
            "CoordinatedResidual_experiment_id",
            "BaiyePowerTargetRates_experiment_id",
            "BaiyePowerRewardRouting_experiment_id",
            "DirectionInvariantObservation_experiment_id",
            "StructuredResidual_experiment_id",
        ):
            if str(getattr(self.config, field, "")) != self.FROZEN_EXPERIMENT_ID:
                raise ValueError("C26-N6 identity mismatch: {}".format(field))
        gain = float(
            getattr(
                self.config,
                "CoordinatedResidual_direct_flight_path_feedback_gain_per_rad",
                math.nan,
            )
        )
        limit = float(
            getattr(
                self.config,
                "CoordinatedResidual_direct_flight_path_feedback_max_action",
                math.nan,
            )
        )
        if (gain, limit) not in self.ALLOWED_GAIN_LIMIT_PAIRS:
            raise ValueError("C26-N6 gain/limit pair is not preregistered")
