"""C21-S1: DAgger state-coverage smoke for the direct student."""

from .baiyepower_direct_student_c21_s0_task import (
    BaiyePowerDirectStudentC21S0Task,
)


class BaiyePowerDirectStudentC21S1Task(BaiyePowerDirectStudentC21S0Task):
    """New identity; deployment still forbids analytic action composition."""

    FROZEN_TASK_NAME = "baiyepower_direct_student_c21_s1"
    FROZEN_EXPERIMENT_ID = "C21-S1"

    def _validate_isolation_config(self):
        super()._validate_isolation_config()
        if bool(getattr(self.config, "DirectStudentTeacher_enable", True)):
            raise ValueError(
                "C21-S1 scenario must default teacher instrumentation off"
            )
