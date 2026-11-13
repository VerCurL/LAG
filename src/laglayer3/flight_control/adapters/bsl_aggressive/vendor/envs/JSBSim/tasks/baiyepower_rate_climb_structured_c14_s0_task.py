"""C14-S0: matched rate/climb curriculum structured-feedback smoke."""

from .baiyepower_structured_feedback_c13_s1_task import (
    BaiyePowerStructuredFeedbackC13S1Task,
)


class BaiyePowerRateClimbStructuredC14S0Task(
    BaiyePowerStructuredFeedbackC13S1Task
):
    """Keep C13 authority while matching the formal rate/climb task."""

    FROZEN_TASK_NAME = "baiyepower_rate_climb_structured_c14_s0"
    FROZEN_EXPERIMENT_ID = "C14-S0"
