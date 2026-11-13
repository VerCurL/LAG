"""C12-D0: read-only extended positive-gain boundary scan identity."""

from .baiyepower_sign_constrained_c11_s0_task import (
    BaiyePowerSignConstrainedC11S0Task,
)


class BaiyePowerExtendedGainC12D0Task(
    BaiyePowerSignConstrainedC11S0Task
):
    """Extend the analytic gain range for fixed-action evaluation only."""

    FROZEN_TASK_NAME = "baiyepower_extended_gain_c12_d0"
    FROZEN_EXPERIMENT_ID = "C12-D0"
    FROZEN_GAIN_FRACTION = 1.0
