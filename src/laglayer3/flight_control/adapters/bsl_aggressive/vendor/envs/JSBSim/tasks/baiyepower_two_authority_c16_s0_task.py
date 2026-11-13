"""C16-S0: effective two-authority structured PPO smoke identity."""

from .baiyepower_three_authority_c15_s0_task import (
    BaiyePowerThreeAuthorityC15S0Task,
)


class BaiyePowerTwoAuthorityC16S0Task(BaiyePowerThreeAuthorityC15S0Task):
    """Keep only course and energy-exchange PPO authorities effective."""

    FROZEN_TASK_NAME = "baiyepower_two_authority_c16_s0"
    FROZEN_EXPERIMENT_ID = "C16-S0"
    FROZEN_ACTION_MASK = (1.0, 0.0, 0.0, 1.0)
