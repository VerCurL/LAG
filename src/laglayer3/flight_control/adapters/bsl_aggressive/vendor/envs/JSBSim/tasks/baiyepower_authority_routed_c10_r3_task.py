"""C10-R3: formal fresh PPO identity for authority-routed reward."""

from .baiyepower_authority_routed_c10_d10_task import (
    BaiyePowerAuthorityRoutedC10D10Task,
)


class BaiyePowerAuthorityRoutedC10R3Task(
    BaiyePowerAuthorityRoutedC10D10Task
):
    """Freeze the D10-passed mechanism for one formal training run."""

    FROZEN_TASK_NAME = "baiyepower_authority_routed_c10_r3"
    FROZEN_EXPERIMENT_ID = "C10-R3"
