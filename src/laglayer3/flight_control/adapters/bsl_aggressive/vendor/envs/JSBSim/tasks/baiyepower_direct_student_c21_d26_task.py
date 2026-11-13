"""C21-D26: frozen analytic-teacher channel-intervention diagnostic."""

import math

import numpy as np

from .baiyepower_direct_student_c21_s0_task import (
    BaiyePowerDirectStudentC21S0Task,
)


class BaiyePowerDirectStudentC21D26Task(BaiyePowerDirectStudentC21S0Task):
    """Permit explicit teacher-channel replacement without changing the task."""

    FROZEN_TASK_NAME = "baiyepower_direct_student_c21_d26"
    FROZEN_EXPERIMENT_ID = "C21-D26"
    ALLOWED_REPLACE_MASKS = {
        (0, 0, 0, 0),
        (1, 0, 0, 0),
        (0, 1, 0, 0),
        (0, 0, 1, 0),
        (0, 0, 0, 1),
        (1, 1, 0, 0),
        (1, 1, 1, 1),
    }

    def _validate_isolation_config(self):
        super()._validate_isolation_config()
        if bool(getattr(self.config, "DirectStudentTeacher_enable", False)):
            raise ValueError("C21-D26 must not enable whole-action teacher mixing")
        if not bool(
            getattr(self.config, "DirectStudentTeacherKL_enable", False)
        ):
            raise ValueError("C21-D26 requires same-state analytic labels")
        fraction = float(
            getattr(self.config, "DirectStudentTeacher_student_fraction", math.nan)
        )
        if not math.isclose(fraction, 1.0, rel_tol=0.0, abs_tol=1e-15):
            raise ValueError("C21-D26 non-replaced channels must be 100% student")
        mask = np.asarray(
            getattr(self.config, "DirectStudentTeacher_replace_mask", []),
            dtype=np.float64,
        ).reshape(-1)
        if mask.shape != (4,) or tuple(int(value) for value in mask) not in self.ALLOWED_REPLACE_MASKS:
            raise ValueError("C21-D26 received an unregistered channel mask")
