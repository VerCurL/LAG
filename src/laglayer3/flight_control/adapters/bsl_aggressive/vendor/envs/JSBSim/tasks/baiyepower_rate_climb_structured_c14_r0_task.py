"""C14-R0: formal fresh-PPO matched rate/climb curriculum identity."""

from .baiyepower_rate_climb_structured_c14_s0_task import (
    BaiyePowerRateClimbStructuredC14S0Task,
)


class BaiyePowerRateClimbStructuredC14R0Task(
    BaiyePowerRateClimbStructuredC14S0Task
):
    """Freeze the C14-S0-passed mechanism for formal training."""

    FROZEN_TASK_NAME = "baiyepower_rate_climb_structured_c14_r0"
    FROZEN_EXPERIMENT_ID = "C14-R0"
