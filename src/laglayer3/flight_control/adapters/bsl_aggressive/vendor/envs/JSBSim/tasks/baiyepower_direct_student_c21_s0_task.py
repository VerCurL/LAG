"""C21-S0: direct-actuator student with the analytic controller disabled."""

import math

from .baiyepower_recentered_residual_c17_r0_task import (
    BaiyePowerRecenteredResidualC17R0Task,
)
from .baiyepower_v2_task import BaiyePowerV2Task


class BaiyePowerDirectStudentC21S0Task(
    BaiyePowerRecenteredResidualC17R0Task
):
    """Keep the C20 plant/task/safety stack but remove analytic action composition."""

    FROZEN_TASK_NAME = "baiyepower_direct_student_c21_s0"
    FROZEN_EXPERIMENT_ID = "C21-S0"

    def _validate_isolation_config(self):
        # The C17 ancestry deliberately requires the analytic residual path.
        # C21 is a new action identity, so retain the common BaiyePowerV2
        # invariants while replacing that requirement with a strict off gate.
        BaiyePowerV2Task._validate_isolation_config(self)
        if str(getattr(self.config, "task", "")) != self.FROZEN_TASK_NAME:
            raise ValueError("C21-S0 requires its isolated direct-student task")
        if bool(getattr(self.config, "CoordinatedResidual_enable", True)):
            raise ValueError("C21-S0 must disable analytic action composition")
        for field in (
            "CoordinatedResidual_experiment_id",
            "BaiyePowerTargetRates_experiment_id",
            "BaiyePowerRewardRouting_experiment_id",
            "DirectionInvariantObservation_experiment_id",
            "StructuredResidual_experiment_id",
        ):
            if str(getattr(self.config, field, "")).strip() != self.FROZEN_EXPERIMENT_ID:
                raise ValueError("{} must be frozen at C21-S0".format(field))
        if str(getattr(self.config, "obs_version", "")).lower() != "v5_target_rates":
            raise ValueError("C21-S0 retains the 32D target-rate observation")
        innovation = float(
            getattr(
                self.config,
                "CoordinatedResidual_energy_trim_max_innovation",
                math.nan,
            )
        )
        if not math.isclose(innovation, 0.05, rel_tol=0.0, abs_tol=1e-15):
            raise ValueError("C21-S0 must inherit the frozen C20-D3 trim identity")
