"""C10-R4: formal fresh-PPO identity for direction-invariant observation."""

from .baiyepower_direction_invariant_c10_s2_task import (
    BaiyePowerDirectionInvariantC10S2Task,
)


class BaiyePowerDirectionInvariantC10R4Task(
    BaiyePowerDirectionInvariantC10S2Task
):
    """Freeze the C10-S2-passed mechanism for one formal run."""

    FROZEN_TASK_NAME = "baiyepower_direction_invariant_c10_r4"
    FROZEN_EXPERIMENT_ID = "C10-R4"
