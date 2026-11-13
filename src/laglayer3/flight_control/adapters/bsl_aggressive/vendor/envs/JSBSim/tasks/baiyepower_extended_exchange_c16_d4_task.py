"""C16-D4: deterministic extended positive exchange-boundary identity."""

from .baiyepower_two_authority_c16_r0_task import (
    BaiyePowerTwoAuthorityC16R0Task,
)


class BaiyePowerExtendedExchangeC16D4Task(BaiyePowerTwoAuthorityC16R0Task):
    """Extend only the exchange multiplier reach for a fixed-action audit."""

    FROZEN_TASK_NAME = "baiyepower_extended_exchange_c16_d4"
    FROZEN_EXPERIMENT_ID = "C16-D4"
    FROZEN_STRUCTURED_EXCHANGE_FRACTION = 0.40
