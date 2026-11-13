"""C21-S4: PPO smoke initialized from the supported signed student."""

from .baiyepower_direct_student_c21_s3_task import (
    BaiyePowerDirectStudentC21S3Task,
)


class BaiyePowerDirectStudentC21S4Task(BaiyePowerDirectStudentC21S3Task):
    """New identity; retain signed observation and direct actuator semantics."""

    FROZEN_TASK_NAME = "baiyepower_direct_student_c21_s4"
    FROZEN_EXPERIMENT_ID = "C21-S4"
