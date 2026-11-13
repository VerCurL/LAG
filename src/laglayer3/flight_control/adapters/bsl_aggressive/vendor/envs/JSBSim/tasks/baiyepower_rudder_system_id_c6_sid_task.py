"""C6-SID probe-only task for the FCS rudder command interface."""

import numpy as np

from .baiyepower_upright_v1_task import BaiyePowerUprightV1Task


class BaiyePowerRudderSystemIdC6SidTask(BaiyePowerUprightV1Task):
    """Permit only a small scripted rudder excitation; never PPO training."""

    FROZEN_RESIDUAL_SCALE = np.asarray(
        [0.0, 0.0, 0.10, 0.0],
        dtype=np.float32,
    )

    def _validate_isolation_config(self):
        super()._validate_isolation_config()
        if str(getattr(self.config, "task", "")) != (
            "baiyepower_rudder_system_id_c6_sid"
        ):
            raise ValueError("C6-SID requires its isolated task name")
        if str(getattr(self.config, "obs_version", "")).lower() != (
            "v4_qp_state"
        ):
            raise ValueError("C6-SID must retain the U1 29D observation")
        if str(
            getattr(self.config, "CoordinatedResidual_mode", "")
        ).strip().lower() != "residual":
            raise ValueError("C6-SID requires residual probe mode")
        if str(
            getattr(
                self.config,
                "CoordinatedResidual_energy_allocator_mode",
                "",
            )
        ).strip().lower() != "adaptive_total_energy_throttle_v1":
            raise ValueError(
                "C6-SID must retain the C5-N2 energy controller"
            )
        if str(
            getattr(self.config, "CoordinatedResidual_experiment_id", "")
        ).strip() != "C6-SID":
            raise ValueError("C6-SID requires experiment id C6-SID")
        if float(
            getattr(
                self.config,
                "CoordinatedResidual_sideslip_rudder_gain_per_deg",
                0.0,
            )
        ) != 0.0:
            raise ValueError(
                "C6-SID forbids closed-loop sideslip rudder feedback"
            )
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
                "C6-SID residual scale is frozen at "
                "[0.0, 0.0, 0.10, 0.0]"
            )
