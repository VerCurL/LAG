"""C16-R0: formal fresh-PPO effective-two-authority identity."""

from .baiyepower_two_authority_c16_s0_task import (
    BaiyePowerTwoAuthorityC16S0Task,
)


class BaiyePowerTwoAuthorityC16R0Task(BaiyePowerTwoAuthorityC16S0Task):
    """Freeze the C16-S0-passed mechanism for formal training."""

    FROZEN_TASK_NAME = "baiyepower_two_authority_c16_r0"
    FROZEN_EXPERIMENT_ID = "C16-R0"
