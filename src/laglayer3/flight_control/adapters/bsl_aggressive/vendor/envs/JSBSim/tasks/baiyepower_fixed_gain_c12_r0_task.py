"""C12-R0: deterministic analytic C7 gain selected by C12-D0."""

import math

from .baiyepower_extended_gain_c12_d0_task import (
    BaiyePowerExtendedGainC12D0Task,
)


class BaiyePowerFixedGainC12R0Task(BaiyePowerExtendedGainC12D0Task):
    """Freeze the smallest C12-D0 factor that passed every gate."""

    FROZEN_TASK_NAME = "baiyepower_fixed_gain_c12_r0"
    FROZEN_EXPERIMENT_ID = "C12-R0"
    FROZEN_MODE = "energy_exchange_fixed_gain"
    FROZEN_FIXED_GAIN = 1.375

    def _validate_isolation_config(self):
        super()._validate_isolation_config()
        fixed_gain = float(
            getattr(
                self.config,
                "CoordinatedResidual_fixed_energy_exchange_gain",
                float("nan"),
            )
        )
        if not math.isfinite(fixed_gain) or not math.isclose(
            fixed_gain,
            self.FROZEN_FIXED_GAIN,
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise ValueError("C12-R0 fixed analytic gain is frozen at 1.375")
