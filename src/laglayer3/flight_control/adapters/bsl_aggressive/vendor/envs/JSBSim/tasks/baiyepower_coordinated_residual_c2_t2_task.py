"""C2-T2 task: throttle-only PPO residual with frozen nominal headroom."""

import math

import numpy as np

from .baiyepower_upright_v1_task import BaiyePowerUprightV1Task


class BaiyePowerCoordinatedResidualC2T2Task(BaiyePowerUprightV1Task):
    """Reduce only the nominal speed-to-throttle gain from C2-T1."""

    FROZEN_RESIDUAL_SCALE = np.asarray(
        [0.0, 0.0, 0.0, 0.40],
        dtype=np.float32,
    )
    FROZEN_SPEED_THROTTLE_GAIN = 0.004

    def _validate_isolation_config(self):
        super()._validate_isolation_config()
        if str(getattr(self.config, "task", "")) != (
            "baiyepower_coordinated_residual_c2_t2"
        ):
            raise ValueError("C2-T2 requires its isolated task name")
        if str(getattr(self.config, "obs_version", "")).lower() != (
            "v4_qp_state"
        ):
            raise ValueError("C2-T2 must retain the U1 29D v4_qp_state")
        if bool(
            getattr(
                self.config,
                "BaiyePowerTargetRates_enable",
                False,
            )
        ):
            raise ValueError(
                "C2-T2 cannot enable target-rate observations"
            )
        if not bool(
            getattr(self.config, "CoordinatedResidual_enable", False)
        ):
            raise ValueError(
                "baiyepower_coordinated_residual_c2_t2 requires "
                "CoordinatedResidual_enable=true"
            )
        mode = str(
            getattr(self.config, "CoordinatedResidual_mode", "")
        ).strip().lower()
        if mode != "residual":
            raise ValueError("C2-T2 requires residual mode")
        if str(
            getattr(self.config, "CoordinatedResidual_rate_source", "")
        ).strip().lower() != "route_analytic_v1":
            raise ValueError(
                "C2-T2 target rates must use route_analytic_v1"
            )
        if str(
            getattr(self.config, "CoordinatedResidual_experiment_id", "")
        ).strip() != "C2-T2":
            raise ValueError("C2-T2 requires experiment id C2-T2")

        residual_scale = np.asarray(
            getattr(
                self.config,
                "CoordinatedResidual_residual_scale",
                [],
            ),
            dtype=np.float32,
        ).reshape(-1)
        if (
            residual_scale.shape != (4,)
            or not np.all(np.isfinite(residual_scale))
            or not np.array_equal(
                residual_scale,
                self.FROZEN_RESIDUAL_SCALE,
            )
        ):
            raise ValueError(
                "C2-T2 residual scale is frozen at "
                "[0.0, 0.0, 0.0, 0.40]"
            )

        speed_throttle_gain = float(
            getattr(
                self.config,
                "CoordinatedResidual_speed_throttle_gain",
                np.nan,
            )
        )
        if (
            not math.isfinite(speed_throttle_gain)
            or not math.isclose(
                speed_throttle_gain,
                self.FROZEN_SPEED_THROTTLE_GAIN,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            raise ValueError(
                "C2-T2 speed-to-throttle gain is frozen at 0.004"
            )
