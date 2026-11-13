"""Isolated C1-R1 task: coordinated nominal control plus bounded PPO residual."""

import numpy as np

from .baiyepower_upright_v1_task import BaiyePowerUprightV1Task


class BaiyePowerCoordinatedResidualC1R1Task(BaiyePowerUprightV1Task):
    """Train only a bounded residual around the frozen C1 nominal controller."""

    FROZEN_RESIDUAL_SCALE = np.asarray(
        [0.20, 0.20, 0.10, 0.40],
        dtype=np.float32,
    )

    def _validate_isolation_config(self):
        super()._validate_isolation_config()
        if str(getattr(self.config, "task", "")) != (
            "baiyepower_coordinated_residual_c1_r1"
        ):
            raise ValueError("C1-R1 requires its isolated task name")
        if str(getattr(self.config, "obs_version", "")).lower() != (
            "v4_qp_state"
        ):
            raise ValueError("C1-R1 must retain the U1 29D v4_qp_state")
        if bool(
            getattr(
                self.config,
                "BaiyePowerTargetRates_enable",
                False,
            )
        ):
            raise ValueError(
                "C1-R1 must not inherit O1 target-rate observations"
            )
        if not bool(
            getattr(self.config, "CoordinatedResidual_enable", False)
        ):
            raise ValueError(
                "baiyepower_coordinated_residual_c1_r1 requires "
                "CoordinatedResidual_enable=true"
            )
        mode = str(
            getattr(self.config, "CoordinatedResidual_mode", "")
        ).strip().lower()
        if mode != "residual":
            raise ValueError("C1-R1 requires residual mode")
        if str(
            getattr(self.config, "CoordinatedResidual_rate_source", "")
        ).strip().lower() != "route_analytic_v1":
            raise ValueError(
                "C1-R1 target rates must use route_analytic_v1"
            )
        if str(
            getattr(self.config, "CoordinatedResidual_experiment_id", "")
        ).strip() != "C1-R1":
            raise ValueError("C1-R1 requires experiment id C1-R1")

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
                "C1-R1 residual scale is frozen at "
                "[0.20, 0.20, 0.10, 0.40]"
            )
