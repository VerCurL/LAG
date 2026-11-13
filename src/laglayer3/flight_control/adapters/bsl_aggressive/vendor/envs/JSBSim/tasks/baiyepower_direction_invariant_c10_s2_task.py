"""C10-S2: direction-invariant observation smoke for the slow PPO action."""

import numpy as np

from .baiyepower_authority_routed_c10_r3_task import (
    BaiyePowerAuthorityRoutedC10R3Task,
)


class BaiyePowerDirectionInvariantC10S2Task(
    BaiyePowerAuthorityRoutedC10R3Task
):
    """Remove left/right sign while retaining turn-related magnitudes."""

    FROZEN_TASK_NAME = "baiyepower_direction_invariant_c10_s2"
    FROZEN_EXPERIMENT_ID = "C10-S2"
    FROZEN_OBSERVATION_MODE = "abs_direction_odd_v1"
    DIRECTION_ODD_INDICES = (1, 4, 9, 12, 14, 17, 19, 21, 23, 25, 27, 29)

    def _validate_isolation_config(self):
        super()._validate_isolation_config()
        mode = str(
            getattr(self.config, "DirectionInvariantObservation_mode", "")
        ).strip().lower()
        if mode != self.FROZEN_OBSERVATION_MODE:
            raise ValueError(
                "C10-S2 observation mode must remain {}".format(
                    self.FROZEN_OBSERVATION_MODE
                )
            )
        experiment_id = str(
            getattr(
                self.config,
                "DirectionInvariantObservation_experiment_id",
                "",
            )
        ).strip()
        if experiment_id != self.FROZEN_EXPERIMENT_ID:
            raise ValueError("C10-S2 requires its observation experiment id")
        indices = tuple(
            int(value)
            for value in getattr(
                self.config,
                "DirectionInvariantObservation_indices",
                (),
            )
        )
        if indices != self.DIRECTION_ODD_INDICES:
            raise ValueError("C10-S2 direction-odd indices are frozen")

    def get_obs(self, env, agent_id):
        observation = np.asarray(
            super().get_obs(env, agent_id),
            dtype=np.float32,
        ).copy()
        if observation.shape != (32,) or not np.all(np.isfinite(observation)):
            raise RuntimeError("C10-S2 requires a finite 32D observation")
        indices = np.asarray(self.DIRECTION_ODD_INDICES, dtype=np.int64)
        observation[indices] = np.abs(observation[indices])
        return observation
