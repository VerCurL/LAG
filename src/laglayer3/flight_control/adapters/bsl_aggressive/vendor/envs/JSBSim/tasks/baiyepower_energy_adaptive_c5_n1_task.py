"""C5-N1 task: online direction-agnostic energy trim estimation."""

import math

import numpy as np

from .baiyepower_upright_v1_task import BaiyePowerUprightV1Task


class BaiyePowerEnergyAdaptiveC5N1Task(BaiyePowerUprightV1Task):
    """Validate online energy-rate bias estimation without PPO."""

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
        "CoordinatedResidual_energy_trim_adaptation_tau_sec": 5.0,
        "CoordinatedResidual_min_adaptive_trim_throttle": 0.35,
        "CoordinatedResidual_max_adaptive_trim_throttle": 0.75,
    }

    def _validate_isolation_config(self):
        super()._validate_isolation_config()
        if str(getattr(self.config, "task", "")) != (
            "baiyepower_energy_adaptive_c5_n1"
        ):
            raise ValueError("C5-N1 requires its isolated task name")
        if str(getattr(self.config, "obs_version", "")).lower() != (
            "v4_qp_state"
        ):
            raise ValueError("C5-N1 must retain the U1 29D observation")
        if bool(
            getattr(
                self.config,
                "BaiyePowerTargetRates_enable",
                False,
            )
        ):
            raise ValueError(
                "C5-N1 cannot enable target-rate observations"
            )
        if str(
            getattr(self.config, "CoordinatedResidual_mode", "")
        ).strip().lower() != "zero_residual":
            raise ValueError("C5-N1 requires zero_residual mode")
        if str(
            getattr(
                self.config,
                "CoordinatedResidual_energy_allocator_mode",
                "",
            )
        ).strip().lower() != "adaptive_total_energy_throttle_v1":
            raise ValueError(
                "C5-N1 requires adaptive_total_energy_throttle_v1"
            )
        if str(
            getattr(self.config, "CoordinatedResidual_experiment_id", "")
        ).strip() != "C5-N1":
            raise ValueError("C5-N1 requires experiment id C5-N1")
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
                "C5-N1 direct residual scales must remain zero"
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
        gain = float(
            self.config.CoordinatedResidual_throttle_energy_rate_gain
        )
        plant_gain = float(
            self.config.CoordinatedResidual_energy_rate_plant_gain
        )
        if not math.isclose(
            gain * plant_gain,
            1.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "C5-N1 plant gain and inverse gain must be reciprocal"
            )
