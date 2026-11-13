"""C16-D6: exact fixed-action exchange-anchor identity."""

from .baiyepower_narrow_exchange_c16_d5_task import (
    BaiyePowerNarrowExchangeC16D5Task,
)


class BaiyePowerExactAnchorC16D6Task(BaiyePowerNarrowExchangeC16D5Task):
    """Remove float32 action quantization from the 1.4 overlap anchor."""

    FROZEN_TASK_NAME = "baiyepower_exact_anchor_c16_d6"
    FROZEN_EXPERIMENT_ID = "C16-D6"
    # 0.4 / float(np.float32(0.8)); multiplying by the fixed float32
    # action 0.8 therefore reproduces the D4 multiplier exactly.
    FROZEN_STRUCTURED_EXCHANGE_FRACTION = 0.4999999925494195
