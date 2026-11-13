"""C5-N2 task: SID2-calibrated online energy trim dynamics."""

import math

import numpy as np

from .baiyepower_upright_v1_task import BaiyePowerUprightV1Task


class BaiyePowerEnergyAdaptiveC5N2Task(BaiyePowerUprightV1Task):
    """Validate the independently identified 3.8 second trim filter."""

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
    }

    def _validate_isolation_config(self):
        super()._validate_isolation_config()
        if str(getattr(self.config, "task", "")) != (
            "baiyepower_energy_adaptive_c5_n2"
        ):
            raise ValueError("C5-N2 requires its isolated task name")
        if str(getattr(self.config, "obs_version", "")).lower() != (
            "v4_qp_state"
        ):
            raise ValueError("C5-N2 must retain the U1 29D observation")
        if bool(
            getattr(
                self.config,
                "BaiyePowerTargetRates_enable",
                False,
            )
        ):
            raise ValueError(
                "C5-N2 cannot enable target-rate observations"
            )
        if str(
            getattr(self.config, "CoordinatedResidual_mode", "")
        ).strip().lower() != "zero_residual":
            raise ValueError("C5-N2 requires zero_residual mode")
        if str(
            getattr(
                self.config,
                "CoordinatedResidual_energy_allocator_mode",
                "",
            )
        ).strip().lower() != "adaptive_total_energy_throttle_v1":
            raise ValueError(
                "C5-N2 requires adaptive_total_energy_throttle_v1"
            )
        if str(
            getattr(self.config, "CoordinatedResidual_experiment_id", "")
        ).strip() != "C5-N2":
            raise ValueError("C5-N2 requires experiment id C5-N2")
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
                "C5-N2 direct residual scales must remain zero"
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
