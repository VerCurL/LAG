"""C7-SID task: probe-only longitudinal energy exchange command."""

import math

import numpy as np

from .baiyepower_upright_v1_task import BaiyePowerUprightV1Task


class BaiyePowerEnergyBalanceSystemIdC7SidTask(
    BaiyePowerUprightV1Task
):
    """Permit only a small scripted flight-path offset; never PPO."""

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
        "CoordinatedResidual_vertical_speed_command_scale_mps": 2.0,
    }

    def _validate_isolation_config(self):
        super()._validate_isolation_config()
        if str(getattr(self.config, "task", "")) != (
            "baiyepower_energy_balance_system_id_c7_sid"
        ):
            raise ValueError("C7-SID requires its isolated task name")
        if str(getattr(self.config, "obs_version", "")).lower() != (
            "v4_qp_state"
        ):
            raise ValueError("C7-SID must retain the U1 29D observation")
        if str(
            getattr(self.config, "CoordinatedResidual_mode", "")
        ).strip().lower() != "energy_balance_probe":
            raise ValueError("C7-SID requires energy_balance_probe mode")
        if str(
            getattr(
                self.config,
                "CoordinatedResidual_energy_allocator_mode",
                "",
            )
        ).strip().lower() != "adaptive_total_energy_throttle_v1":
            raise ValueError(
                "C7-SID must retain adaptive_total_energy_throttle_v1"
            )
        if str(
            getattr(self.config, "CoordinatedResidual_experiment_id", "")
        ).strip() != "C7-SID":
            raise ValueError("C7-SID requires experiment id C7-SID")
        action_index = getattr(
            self.config,
            "CoordinatedResidual_command_action_index",
            None,
        )
        if isinstance(action_index, bool) or action_index != 3:
            raise ValueError("C7-SID command action index is frozen at 3")
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
                "C7-SID direct residual scales must remain zero"
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
