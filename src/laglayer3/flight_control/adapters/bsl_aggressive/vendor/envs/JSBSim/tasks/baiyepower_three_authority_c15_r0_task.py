"""C15-R0: formal fresh-PPO effective-three-authority identity."""

from .baiyepower_three_authority_c15_s0_task import (
    BaiyePowerThreeAuthorityC15S0Task,
)


class BaiyePowerThreeAuthorityC15R0Task(
    BaiyePowerThreeAuthorityC15S0Task
):
    """Freeze the C15-S0-passed mechanism for formal training."""

    FROZEN_TASK_NAME = "baiyepower_three_authority_c15_r0"
    FROZEN_EXPERIMENT_ID = "C15-R0"
