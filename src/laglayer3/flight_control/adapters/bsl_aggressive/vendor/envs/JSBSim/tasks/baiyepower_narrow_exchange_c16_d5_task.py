"""C16-D5: narrow deterministic exchange-boundary identity."""

from .baiyepower_extended_exchange_c16_d4_task import (
    BaiyePowerExtendedExchangeC16D4Task,
)


class BaiyePowerNarrowExchangeC16D5Task(
    BaiyePowerExtendedExchangeC16D4Task
):
    """Scan the legal upper exchange boundary for a fixed-action audit."""

    FROZEN_TASK_NAME = "baiyepower_narrow_exchange_c16_d5"
    FROZEN_EXPERIMENT_ID = "C16-D5"
    FROZEN_STRUCTURED_EXCHANGE_FRACTION = 0.50
