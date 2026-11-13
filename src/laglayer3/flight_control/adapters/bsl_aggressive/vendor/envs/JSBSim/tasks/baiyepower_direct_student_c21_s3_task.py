"""C21-S3: signed observation for a direct actuator student."""

import numpy as np

from .baiyepower_direct_student_c21_s2_task import (
    BaiyePowerDirectStudentC21S2Task,
)
from .heading_task import HeadingTask


class BaiyePowerDirectStudentC21S3Task(BaiyePowerDirectStudentC21S2Task):
    """Restore direction-odd signs that direct lateral control requires."""

    FROZEN_TASK_NAME = "baiyepower_direct_student_c21_s3"
    FROZEN_EXPERIMENT_ID = "C21-S3"
    FROZEN_OBSERVATION_MODE = "signed_direct_actuator_v1"

    def _validate_isolation_config(self):
        super()._validate_isolation_config()
        mode = str(
            getattr(self.config, "DirectionInvariantObservation_mode", "")
        ).strip().lower()
        if mode != self.FROZEN_OBSERVATION_MODE:
            raise ValueError("C21-S3 requires signed direct-actuator observation")

    def get_obs(self, env, agent_id):
        # The C11-C21 ancestry canonicalized direction-odd state because its
        # policy only selected scalar gains around an analytic controller.
        # A direct A/E/R/T actor needs the physical signs, so call the common
        # 32D HeadingTask observation before that canonicalization layer.
        observation = np.asarray(
            HeadingTask.get_obs(self, env, agent_id), dtype=np.float32
        )
        if observation.shape != (32,) or not np.all(np.isfinite(observation)):
            raise RuntimeError("C21-S3 requires a finite signed 32D observation")
        return observation
