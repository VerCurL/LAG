"""C26-SID1: confirm the fast-q / slow-flight-path response split."""

from .baiyepower_pitch_response_c26_sid0_task import (
    BaiyePowerPitchResponseC26SID0Task,
)


class BaiyePowerPitchResponseC26SID1Task(
    BaiyePowerPitchResponseC26SID0Task
):
    """Keep SID0 dynamics and authority under a distinct audit identity."""

    FROZEN_TASK_NAME = "baiyepower_pitch_response_c26_sid1"
    FROZEN_EXPERIMENT_ID = "C26-SID1"
