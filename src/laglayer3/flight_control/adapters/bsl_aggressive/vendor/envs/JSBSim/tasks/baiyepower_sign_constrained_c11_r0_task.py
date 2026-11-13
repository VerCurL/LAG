"""C11-R0: formal fresh-PPO identity for the sign-constrained gain residual."""

from .baiyepower_sign_constrained_c11_s0_task import (
    BaiyePowerSignConstrainedC11S0Task,
)


class BaiyePowerSignConstrainedC11R0Task(
    BaiyePowerSignConstrainedC11S0Task
):
    """Freeze the C11-S0-passed mechanism for one formal run."""

    FROZEN_TASK_NAME = "baiyepower_sign_constrained_c11_r0"
    FROZEN_EXPERIMENT_ID = "C11-R0"
