"""C21-S5: low-exploration PPO smoke from the supported signed student."""

from .baiyepower_direct_student_c21_s4_task import (
    BaiyePowerDirectStudentC21S4Task,
)


class BaiyePowerDirectStudentC21S5Task(BaiyePowerDirectStudentC21S4Task):
    """New identity; isolate the PPO exploration standard deviation."""

    FROZEN_TASK_NAME = "baiyepower_direct_student_c21_s5"
    FROZEN_EXPERIMENT_ID = "C21-S5"
