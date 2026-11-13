"""C13-S1: point-route-active structured PPO feedback gains."""

import numpy as np

from .baiyepower_structured_residual_c13_s0_task import (
    BaiyePowerStructuredResidualC13S0Task,
)


class BaiyePowerStructuredFeedbackC13S1Task(
    BaiyePowerStructuredResidualC13S0Task
):
    """Expose positive analytic feedback gains, never raw actuators."""

    FROZEN_TASK_NAME = "baiyepower_structured_feedback_c13_s1"
    FROZEN_EXPERIMENT_ID = "C13-S1"
    FROZEN_MODE = "structured_feedback_gain_residual"
    FROZEN_FEEDBACK_GAIN_FRACTIONS = (0.10, 0.15, 0.15)

    def _validate_isolation_config(self):
        super()._validate_isolation_config()
        fractions = np.asarray(
            getattr(
                self.config,
                "CoordinatedResidual_structured_feedback_gain_fractions",
                (),
            ),
            dtype=np.float64,
        ).reshape(-1)
        expected = np.asarray(
            self.FROZEN_FEEDBACK_GAIN_FRACTIONS,
            dtype=np.float64,
        )
        if fractions.shape != (3,) or not np.allclose(
            fractions,
            expected,
            rtol=0.0,
            atol=1e-15,
        ):
            raise ValueError(
                "C13-S1 feedback fractions are frozen at "
                "[0.10, 0.15, 0.15]"
            )
