"""C10-R1: formal slow-command PPO on the frozen C10-S0 control stack."""

from .baiyepower_energy_command_c10_s0_task import (
    BaiyePowerEnergyCommandC10S0Task,
)


class BaiyePowerEnergyCommandC10R1Task(
    BaiyePowerEnergyCommandC10S0Task
):
    """Give the smoke-validated mechanism an independent formal identity."""

    FROZEN_TASK_NAME = "baiyepower_energy_command_c10_r1"
    FROZEN_EXPERIMENT_ID = "C10-R1"
