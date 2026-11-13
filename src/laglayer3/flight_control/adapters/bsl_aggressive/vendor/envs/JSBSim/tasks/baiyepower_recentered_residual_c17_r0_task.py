"""C17-R0: formal fresh PPO around the supported analytic center."""

from .baiyepower_recentered_residual_c17_s0_task import (
    BaiyePowerRecenteredResidualC17S0Task,
)


class BaiyePowerRecenteredResidualC17R0Task(
    BaiyePowerRecenteredResidualC17S0Task
):
    """Freeze the C17-S0-passed mechanism for formal training."""

    FROZEN_TASK_NAME = "baiyepower_recentered_residual_c17_r0"
    FROZEN_EXPERIMENT_ID = "C17-R0"
