"""C16-V1: unseen-seed holdout for the exact 1.410 candidate."""

from .baiyepower_exact_anchor_c16_d6_task import (
    BaiyePowerExactAnchorC16D6Task,
)


class BaiyePowerExchangeHoldoutC16V1Task(BaiyePowerExactAnchorC16D6Task):
    """Freeze the D6 controller while changing only evaluation seeds."""

    FROZEN_TASK_NAME = "baiyepower_exchange_holdout_c16_v1"
    FROZEN_EXPERIMENT_ID = "C16-V1"
