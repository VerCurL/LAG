"""C8-SID isolated aileron/flaperon coupling probe task."""

import math

import numpy as np

from .baiyepower_sideslip_control_c6_r1_task import (
    BaiyePowerSideslipControlC6R1Task,
)
from .baiyepower_upright_v1_task import BaiyePowerUprightV1Task


class BaiyePowerAileronFlaperonSystemIdC8SidTask(
    BaiyePowerUprightV1Task
):
    """Permit only a small scripted aileron excitation; never PPO training."""

    FROZEN_RESIDUAL_SCALE = np.asarray(
        [0.10, 0.0, 0.0, 0.0],
        dtype=np.float32,
    )

    def _validate_isolation_config(self):
        super()._validate_isolation_config()
        if str(getattr(self.config, "task", "")) != (
            "baiyepower_aileron_flaperon_system_id_c8_sid"
        ):
            raise ValueError("C8-SID requires its isolated task name")
        if str(getattr(self.config, "obs_version", "")).lower() != (
            "v4_qp_state"
        ):
            raise ValueError("C8-SID must retain the U1 29D observation")
        if str(
            getattr(self.config, "CoordinatedResidual_mode", "")
        ).strip().lower() != "residual":
            raise ValueError("C8-SID requires residual probe mode")
        if str(
            getattr(
                self.config,
                "CoordinatedResidual_energy_allocator_mode",
                "",
            )
        ).strip().lower() != "adaptive_total_energy_throttle_v1":
            raise ValueError(
                "C8-SID must retain the C6-R1 energy controller"
            )
        if str(
            getattr(self.config, "CoordinatedResidual_experiment_id", "")
        ).strip() != "C8-SID":
            raise ValueError("C8-SID requires experiment id C8-SID")
        scale = np.asarray(
            getattr(
                self.config,
                "CoordinatedResidual_residual_scale",
                [],
            ),
            dtype=np.float32,
        ).reshape(-1)
        if (
            scale.shape != (4,)
            or not np.array_equal(scale, self.FROZEN_RESIDUAL_SCALE)
        ):
            raise ValueError(
                "C8-SID residual scale is frozen at "
                "[0.10, 0.0, 0.0, 0.0]"
            )
        for name, expected in (
            BaiyePowerSideslipControlC6R1Task.FROZEN_VALUES.items()
        ):
            actual = float(getattr(self.config, name, np.nan))
            if (
                not math.isfinite(actual)
                or not math.isclose(
                    actual,
                    expected,
                    rel_tol=0.0,
                    abs_tol=1e-15,
                )
            ):
                raise ValueError(
                    "{} is frozen at {}".format(name, expected)
                )
