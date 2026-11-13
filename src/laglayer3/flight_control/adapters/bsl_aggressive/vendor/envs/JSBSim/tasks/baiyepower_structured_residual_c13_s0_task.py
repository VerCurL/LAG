"""C13-S0: structured PPO gains around the frozen C12 controller."""

import math

import numpy as np

from .baiyepower_fixed_gain_c12_r1_task import (
    BaiyePowerFixedGainC12R1Task,
)


class BaiyePowerStructuredResidualC13S0Task(
    BaiyePowerFixedGainC12R1Task
):
    """Expose sign-preserving reference gains, never raw actuator residuals."""

    FROZEN_TASK_NAME = "baiyepower_structured_residual_c13_s0"
    FROZEN_EXPERIMENT_ID = "C13-S0"
    FROZEN_MODE = "structured_reference_gain_residual"
    FROZEN_REWARD_ROUTING_MODE = "full_structured_control_v1"
    FROZEN_REFERENCE_GAIN_FRACTIONS = (0.10, 0.15, 0.15)
    FROZEN_STRUCTURED_EXCHANGE_FRACTION = 0.10
    FROZEN_SCHEDULE = {
        "CoordinatedResidual_flight_path_tau_low_speed_sec": 0.90,
        "CoordinatedResidual_flight_path_tau_high_speed_low_alt_sec": 0.97,
        "CoordinatedResidual_flight_path_tau_high_speed_high_alt_sec": 1.00,
        "CoordinatedResidual_flight_path_tau_speed_blend_start_mps": 240.0,
        "CoordinatedResidual_flight_path_tau_speed_blend_end_mps": 260.0,
        "CoordinatedResidual_flight_path_tau_altitude_blend_start_m": 5500.0,
        "CoordinatedResidual_flight_path_tau_altitude_blend_end_m": 6500.0,
    }

    def _validate_isolation_config(self):
        super()._validate_isolation_config()
        fractions = np.asarray(
            getattr(
                self.config,
                "CoordinatedResidual_structured_reference_gain_fractions",
                (),
            ),
            dtype=np.float64,
        ).reshape(-1)
        expected_fractions = np.asarray(
            self.FROZEN_REFERENCE_GAIN_FRACTIONS,
            dtype=np.float64,
        )
        if fractions.shape != (3,) or not np.allclose(
            fractions,
            expected_fractions,
            rtol=0.0,
            atol=1e-15,
        ):
            raise ValueError(
                "C13-S0 structured reference fractions are frozen at "
                "[0.10, 0.15, 0.15]"
            )
        exchange_fraction = float(
            getattr(
                self.config,
                "CoordinatedResidual_structured_energy_exchange_gain_fraction",
                np.nan,
            )
        )
        if not math.isclose(
            exchange_fraction,
            self.FROZEN_STRUCTURED_EXCHANGE_FRACTION,
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise ValueError(
                "C13-S0 structured exchange fraction is frozen at 0.10"
            )
        if not bool(
            getattr(
                self.config,
                "CoordinatedResidual_flight_path_tau_schedule_enable",
                False,
            )
        ):
            raise ValueError("C13-S0 requires the C12 tau schedule")
        for name, expected in self.FROZEN_SCHEDULE.items():
            actual = float(getattr(self.config, name, np.nan))
            if not math.isclose(
                actual,
                expected,
                rel_tol=0.0,
                abs_tol=1e-15,
            ):
                raise ValueError("{} is frozen at {}".format(name, expected))
        residual_scale = np.asarray(
            getattr(self.config, "CoordinatedResidual_residual_scale", ()),
            dtype=np.float64,
        ).reshape(-1)
        if residual_scale.shape != (4,) or np.any(residual_scale != 0.0):
            raise ValueError("C13-S0 direct actuator residual must remain zero")

    def get_reward(self, env, agent_id, info=None):
        routed_reward, info = super().get_reward(env, agent_id, info)
        diag = self._transition_diag.setdefault(agent_id, {})
        direction_reward = float(
            diag.get("direction_reward_routed_out", 0.0)
        )
        full_reward = float(routed_reward) + direction_reward
        if not math.isfinite(full_reward):
            raise RuntimeError("C13-S0 full structured reward is non-finite")
        diag["direction_reward_restored"] = direction_reward
        diag["full_structured_reward_active"] = 1.0
        diag["reward_contribution_sum"] = full_reward
        return full_reward, info
