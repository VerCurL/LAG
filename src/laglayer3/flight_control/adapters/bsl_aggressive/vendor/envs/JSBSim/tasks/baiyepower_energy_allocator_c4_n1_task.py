"""C4-N1 task: total-energy throttle with the frozen C1 altitude loop."""

import math

import numpy as np

from .baiyepower_upright_v1_task import BaiyePowerUprightV1Task


class BaiyePowerEnergyAllocatorC4N1Task(BaiyePowerUprightV1Task):
    """Ablate the failed C4-N0 energy-balance pitch channel."""

    FROZEN_VALUES = {
        "CoordinatedResidual_total_energy_tau_sec": 20.0,
        "CoordinatedResidual_energy_balance_tau_sec": 20.0,
        "CoordinatedResidual_throttle_energy_rate_gain": 0.008,
        "CoordinatedResidual_trim_physical_throttle": 0.65,
        "CoordinatedResidual_policy_throttle_center": 0.65,
        "CoordinatedResidual_policy_throttle_half_range": 0.25,
        "CoordinatedResidual_max_balance_correction_mps": 10.0,
    }

    def _validate_isolation_config(self):
        super()._validate_isolation_config()
        if str(getattr(self.config, "task", "")) != (
            "baiyepower_energy_allocator_c4_n1"
        ):
            raise ValueError("C4-N1 requires its isolated task name")
        if str(getattr(self.config, "obs_version", "")).lower() != (
            "v4_qp_state"
        ):
            raise ValueError("C4-N1 must retain the U1 29D observation")
        if bool(
            getattr(
                self.config,
                "BaiyePowerTargetRates_enable",
                False,
            )
        ):
            raise ValueError(
                "C4-N1 cannot enable target-rate observations"
            )
        if not bool(
            getattr(self.config, "CoordinatedResidual_enable", False)
        ):
            raise ValueError(
                "C4-N1 requires CoordinatedResidual_enable=true"
            )
        if str(
            getattr(self.config, "CoordinatedResidual_mode", "")
        ).strip().lower() != "zero_residual":
            raise ValueError("C4-N1 requires zero_residual mode")
        if str(
            getattr(self.config, "CoordinatedResidual_rate_source", "")
        ).strip().lower() != "route_analytic_v1":
            raise ValueError(
                "C4-N1 target rates must use route_analytic_v1"
            )
        if str(
            getattr(
                self.config,
                "CoordinatedResidual_energy_allocator_mode",
                "",
            )
        ).strip().lower() != "total_energy_throttle_v1":
            raise ValueError(
                "C4-N1 requires total_energy_throttle_v1"
            )
        if str(
            getattr(self.config, "CoordinatedResidual_experiment_id", "")
        ).strip() != "C4-N1":
            raise ValueError("C4-N1 requires experiment id C4-N1")

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
                "C4-N1 direct residual scales must remain zero"
            )

        for name, expected in self.FROZEN_VALUES.items():
            actual = float(getattr(self.config, name, np.nan))
            if (
                not math.isfinite(actual)
                or not math.isclose(
                    actual,
                    expected,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            ):
                raise ValueError(
                    "{} is frozen at {}".format(name, expected)
                )
