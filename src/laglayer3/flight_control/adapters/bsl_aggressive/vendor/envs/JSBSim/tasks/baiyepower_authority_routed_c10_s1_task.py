"""C10-S1: fresh PPO engineering smoke for authority-routed reward."""

from .baiyepower_authority_routed_c10_d10_task import (
    BaiyePowerAuthorityRoutedC10D10Task,
)


class BaiyePowerAuthorityRoutedC10S1Task(
    BaiyePowerAuthorityRoutedC10D10Task
):
    """Give the D10-passed mechanism an independent smoke identity."""

    FROZEN_TASK_NAME = "baiyepower_authority_routed_c10_s1"
    FROZEN_EXPERIMENT_ID = "C10-S1"
