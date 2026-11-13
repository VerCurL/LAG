"""C21-S7: micro-step actor-rate PPO smoke from the signed student."""

from .baiyepower_direct_student_c21_s6_task import (
    BaiyePowerDirectStudentC21S6Task,
)


class BaiyePowerDirectStudentC21S7Task(BaiyePowerDirectStudentC21S6Task):
    """New identity; isolate a further tenfold actor-rate reduction."""

    FROZEN_TASK_NAME = "baiyepower_direct_student_c21_s7"
    FROZEN_EXPERIMENT_ID = "C21-S7"
