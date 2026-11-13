"""C7-R0 task: measured-gain, opposite-sign energy exchange."""

import math

import numpy as np

from .baiyepower_upright_v1_task import BaiyePowerUprightV1Task


class BaiyePowerEnergyExchangeC7R0Task(BaiyePowerUprightV1Task):
    """Validate bounded longitudinal energy exchange without PPO."""

    FROZEN_VALUES = {
        "CoordinatedResidual_total_energy_tau_sec": 20.0,
        "CoordinatedResidual_throttle_energy_rate_gain": (
            0.0031909191384566376
        ),
        "CoordinatedResidual_trim_physical_throttle": (
            0.4491547582071449
        ),
        "CoordinatedResidual_energy_rate_plant_gain": (
            313.38932658872494
        ),
        "CoordinatedResidual_energy_trim_adaptation_tau_sec": 3.8,
        "CoordinatedResidual_min_adaptive_trim_throttle": 0.35,
        "CoordinatedResidual_max_adaptive_trim_throttle": 0.75,
        "CoordinatedResidual_sideslip_rudder_gain_per_deg": -0.05,
        "CoordinatedResidual_max_rudder_action": 0.20,
        "CoordinatedResidual_energy_exchange_tau_sec": 60.0,
        "CoordinatedResidual_energy_exchange_height_rate_gain": (
            0.9042178500025742
        ),
        "CoordinatedResidual_energy_exchange_kinetic_rate_gain": (
            0.6609097763737128
        ),
        "CoordinatedResidual_max_energy_exchange_offset_mps": 1.0,
    }

    def _validate_isolation_config(self):
        super()._validate_isolation_config()
        if str(getattr(self.config, "task", "")) != (
            "baiyepower_energy_exchange_c7_r0"
        ):
            raise ValueError("C7-R0 requires its isolated task name")
        if str(getattr(self.config, "obs_version", "")).lower() != (
            "v4_qp_state"
        ):
            raise ValueError("C7-R0 must retain the U1 29D observation")
        if str(
            getattr(self.config, "CoordinatedResidual_mode", "")
        ).strip().lower() != "zero_residual":
            raise ValueError("C7-R0 requires zero_residual mode")
        if str(
            getattr(
                self.config,
                "CoordinatedResidual_energy_allocator_mode",
                "",
            )
        ).strip().lower() != "adaptive_total_energy_throttle_v1":
            raise ValueError(
                "C7-R0 must retain adaptive_total_energy_throttle_v1"
            )
        if str(
            getattr(
                self.config,
                "CoordinatedResidual_energy_exchange_mode",
                "",
            )
        ).strip().lower() != "measured_opposite_sign_v1":
            raise ValueError(
                "C7-R0 requires measured_opposite_sign_v1"
            )
        if str(
            getattr(self.config, "CoordinatedResidual_experiment_id", "")
        ).strip() != "C7-R0":
            raise ValueError("C7-R0 requires experiment id C7-R0")
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
            or not np.array_equal(
                residual_scale,
                np.zeros(4, dtype=np.float32),
            )
        ):
            raise ValueError(
                "C7-R0 direct residual scales must remain zero"
            )
        for name, expected in self.FROZEN_VALUES.items():
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
