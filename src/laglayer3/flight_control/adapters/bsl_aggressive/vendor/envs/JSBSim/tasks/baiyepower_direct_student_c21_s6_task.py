"""C21-S6: conservative actor-rate PPO smoke from the signed student."""

from .baiyepower_direct_student_c21_s4_task import (
    BaiyePowerDirectStudentC21S4Task,
)


class BaiyePowerDirectStudentC21S6Task(BaiyePowerDirectStudentC21S4Task):
    """New identity; isolate actor learning rate from critic learning rate."""

    FROZEN_TASK_NAME = "baiyepower_direct_student_c21_s6"
    FROZEN_EXPERIMENT_ID = "C21-S6"
