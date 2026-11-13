"""C21-S17: training-only analytic-teacher KL for the direct student."""

import math

from .baiyepower_direct_student_c21_s7_task import (
    BaiyePowerDirectStudentC21S7Task,
)


class BaiyePowerDirectStudentC21S17Task(BaiyePowerDirectStudentC21S7Task):
    """Keep direct execution while exposing same-state analytic labels."""

    FROZEN_TASK_NAME = "baiyepower_direct_student_c21_s17"
    FROZEN_EXPERIMENT_ID = "C21-S17"

    def _validate_isolation_config(self):
        super()._validate_isolation_config()
        if not bool(getattr(self.config, "DirectStudentTeacherKL_enable", False)):
            raise ValueError("C21-S17 requires training-only teacher KL labels")
        fraction = float(
            getattr(self.config, "DirectStudentTeacher_student_fraction", math.nan)
        )
        if not math.isclose(fraction, 1.0, rel_tol=0.0, abs_tol=1e-15):
            raise ValueError("C21-S17 must execute 100% student action")
