"""C13-R0: formal fresh-PPO structured feedback identity."""

from .baiyepower_structured_feedback_c13_s1_task import (
    BaiyePowerStructuredFeedbackC13S1Task,
)


class BaiyePowerStructuredFeedbackC13R0Task(
    BaiyePowerStructuredFeedbackC13S1Task
):
    """Freeze the S1-passed four-axis mechanism for formal training."""

    FROZEN_TASK_NAME = "baiyepower_structured_feedback_c13_r0"
    FROZEN_EXPERIMENT_ID = "C13-R0"
