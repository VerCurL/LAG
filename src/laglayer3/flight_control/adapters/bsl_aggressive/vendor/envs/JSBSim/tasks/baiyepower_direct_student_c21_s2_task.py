"""C21-S2: stratified aggregate refit for the direct student."""

from .baiyepower_direct_student_c21_s1_task import (
    BaiyePowerDirectStudentC21S1Task,
)


class BaiyePowerDirectStudentC21S2Task(BaiyePowerDirectStudentC21S1Task):
    """New identity; deployment remains a fully direct student action path."""

    FROZEN_TASK_NAME = "baiyepower_direct_student_c21_s2"
    FROZEN_EXPERIMENT_ID = "C21-S2"
