"""C11-S0: sign-preserving gain residual around the frozen C7 energy law."""

import math

import numpy as np

from .baiyepower_authority_routed_c10_d10_task import (
    BaiyePowerAuthorityRoutedC10D10Task,
)


class BaiyePowerSignConstrainedC11S0Task(
    BaiyePowerAuthorityRoutedC10D10Task
):
    """Expose only a bounded gain modulation of C7's analytic exchange."""

    FROZEN_TASK_NAME = "baiyepower_sign_constrained_c11_s0"
    FROZEN_EXPERIMENT_ID = "C11-S0"
    FROZEN_MODE = "energy_exchange_gain_residual"
    FROZEN_OBSERVATION_MODE = "abs_direction_odd_v1"
    FROZEN_GAIN_FRACTION = 0.25
    DIRECTION_ODD_INDICES = (1, 4, 9, 12, 14, 17, 19, 21, 23, 25, 27, 29)

    def _validate_isolation_config(self):
        super()._validate_isolation_config()
        experiment_id = self.FROZEN_EXPERIMENT_ID
        mode = str(
            getattr(self.config, "DirectionInvariantObservation_mode", "")
        ).strip().lower()
        if mode != self.FROZEN_OBSERVATION_MODE:
            raise ValueError(
                "{} observation mode must remain {}".format(
                    experiment_id, self.FROZEN_OBSERVATION_MODE
                )
            )
        if str(
            getattr(
                self.config,
                "DirectionInvariantObservation_experiment_id",
                "",
            )
        ).strip() != experiment_id:
            raise ValueError("C11-S0 requires its observation experiment id")
        indices = tuple(
            int(value)
            for value in getattr(
                self.config, "DirectionInvariantObservation_indices", ()
            )
        )
        if indices != self.DIRECTION_ODD_INDICES:
            raise ValueError("C11-S0 direction-odd indices are frozen")
        fraction = float(
            getattr(
                self.config,
                "CoordinatedResidual_energy_exchange_gain_fraction",
                np.nan,
            )
        )
        if not math.isfinite(fraction) or not math.isclose(
            fraction,
            self.FROZEN_GAIN_FRACTION,
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise ValueError("C11-S0 gain fraction is frozen at 0.25")

    def get_obs(self, env, agent_id):
        observation = np.asarray(
            super().get_obs(env, agent_id), dtype=np.float32
        ).copy()
        if observation.shape != (32,) or not np.all(np.isfinite(observation)):
            raise RuntimeError("C11-S0 requires a finite 32D observation")
        indices = np.asarray(self.DIRECTION_ODD_INDICES, dtype=np.int64)
        observation[indices] = np.abs(observation[indices])
        return observation
