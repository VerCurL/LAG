"""C12-R1: signed-vertical deterministic gain selected by C12-D1."""

from .baiyepower_fixed_gain_c12_r0_task import (
    BaiyePowerFixedGainC12R0Task,
)


class BaiyePowerFixedGainC12R1Task(BaiyePowerFixedGainC12R0Task):
    """Freeze the smallest factor passing the signed 18-case grid."""

    FROZEN_TASK_NAME = "baiyepower_fixed_gain_c12_r1"
    FROZEN_EXPERIMENT_ID = "C12-R1"
    FROZEN_FIXED_GAIN = 1.4
