"""C15-S0: effective three-authority structured PPO smoke identity."""

import numpy as np

from .baiyepower_rate_climb_structured_c14_s0_task import (
    BaiyePowerRateClimbStructuredC14S0Task,
)


class BaiyePowerThreeAuthorityC15S0Task(
    BaiyePowerRateClimbStructuredC14S0Task
):
    """Disable only the holdout-confirmed harmful altitude gain axis."""

    FROZEN_TASK_NAME = "baiyepower_three_authority_c15_s0"
    FROZEN_EXPERIMENT_ID = "C15-S0"
    FROZEN_ACTION_MASK = (1.0, 0.0, 1.0, 1.0)

    def _validate_isolation_config(self):
        super()._validate_isolation_config()
        mask = np.asarray(
            getattr(
                self.config,
                "CoordinatedResidual_structured_feedback_action_mask",
                (),
            ),
            dtype=np.float64,
        ).reshape(-1)
        expected = np.asarray(self.FROZEN_ACTION_MASK, dtype=np.float64)
        if mask.shape != (4,) or not np.array_equal(mask, expected):
            raise ValueError(
                "C15-S0 structured authority mask is frozen at "
                "[1, 0, 1, 1]"
            )
