"""C17-S0: PPO residuals around the holdout-supported analytic center."""

import math

from .baiyepower_two_authority_c16_s0_task import (
    BaiyePowerTwoAuthorityC16S0Task,
)


class BaiyePowerRecenteredResidualC17S0Task(BaiyePowerTwoAuthorityC16S0Task):
    """Fold C16-V1's fixed candidate into zero-action analytic parity."""

    FROZEN_TASK_NAME = "baiyepower_recentered_residual_c17_s0"
    FROZEN_EXPERIMENT_ID = "C17-S0"
    FROZEN_FIXED_GAIN = 1.9739999864399433
    FROZEN_COURSE_TAU_SEC = 9.09090909090909
    FROZEN_FEEDBACK_GAIN_FRACTIONS = (0.03, 0.15, 0.15)
    FROZEN_STRUCTURED_EXCHANGE_FRACTION = 0.0125

    def _validate_isolation_config(self):
        super()._validate_isolation_config()
        course_tau = float(
            getattr(self.config, "CoordinatedResidual_course_tau_sec", math.nan)
        )
        if not math.isclose(
            course_tau,
            self.FROZEN_COURSE_TAU_SEC,
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise ValueError("C17-S0 course tau is not recentered at 10/1.1")
        maximum_gain = self.FROZEN_FIXED_GAIN * (
            1.0 + self.FROZEN_STRUCTURED_EXCHANGE_FRACTION
        )
        if maximum_gain > 2.0:
            raise ValueError("C17-S0 exchange authority exceeds hard gain cap")
