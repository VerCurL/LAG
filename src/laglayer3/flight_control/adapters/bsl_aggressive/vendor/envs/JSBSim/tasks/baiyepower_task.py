import numpy as np

from .heading_task import HeadingTask
from ..core.catalog import Catalog as c
from ..reward_functions import BaiyePowerEnergyDiagnostic
from ..termination_conditions import BaiyePowerUnreachHeading
from ..utils.baiyepower_physics import (
    STANDARD_GRAVITY_MPS2,
    compute_baiyepower_energy_terms,
    speed_magnitude_mps,
)


class BaiyePowerTask(HeadingTask):
    """Opt-in heading-control task with one scalar speed-magnitude semantics.

    This task deliberately remains a ``HeadingTask`` subclass so the existing
    QP, minimum-jerk and metrics infrastructure stays available.  Its target
    speed, observation slot 2, route completion and active speed tracking all
    use ``sqrt(u**2 + v**2 + w**2)``.  The energy sidecar is diagnostic-only in
    this first frozen revision.
    """

    def __init__(self, config):
        super().__init__(config)
        self.speed_error_scale_mps = float(
            getattr(config, "BaiyePower_speed_error_scale_mps", 30.0)
        )
        self.gravity_mps2 = float(
            getattr(
                config,
                "BaiyePower_gravity_mps2",
                STANDARD_GRAVITY_MPS2,
            )
        )
        if self.speed_error_scale_mps <= 0.0:
            raise ValueError("BaiyePower_speed_error_scale_mps must be positive")
        if self.gravity_mps2 <= 0.0:
            raise ValueError("BaiyePower_gravity_mps2 must be positive")

        self.termination_conditions[0] = BaiyePowerUnreachHeading(config)
        self.reward_functions.append(BaiyePowerEnergyDiagnostic(config))

    @staticmethod
    def get_current_speed_mps(sim):
        return speed_magnitude_mps(
            sim.get_property_value(c.velocities_u_mps),
            sim.get_property_value(c.velocities_v_mps),
            sim.get_property_value(c.velocities_w_mps),
        )

    @staticmethod
    def get_target_speed_mps(sim):
        return float(sim.get_property_value(c.baiyepower_target_speed_mps))

    @staticmethod
    def set_target_speed_mps(sim, target_speed_mps):
        sim.set_property_value(
            c.baiyepower_target_speed_mps,
            max(float(target_speed_mps), 0.0),
        )

    def get_speed_error_mps(self, sim):
        return self.get_target_speed_mps(sim) - self.get_current_speed_mps(sim)

    @staticmethod
    def get_vt_audit_mps(sim):
        """JSBSim true airspeed, used only to audit the no-wind assumption."""

        return float(sim.get_property_value(c.velocities_vt_fps)) * 0.3048

    def get_baiyepower_energy_terms(self, sim):
        return compute_baiyepower_energy_terms(
            altitude_m=sim.get_property_value(c.position_h_sl_m),
            target_altitude_m=(
                sim.get_property_value(c.target_altitude_ft) * 0.3048
            ),
            u_mps=sim.get_property_value(c.velocities_u_mps),
            v_mps=sim.get_property_value(c.velocities_v_mps),
            w_mps=sim.get_property_value(c.velocities_w_mps),
            target_speed_mps=self.get_target_speed_mps(sim),
            gravity_mps2=self.gravity_mps2,
        )

    def reset(self, env):
        # The target must be valid before reward sidecars are reset and before
        # the first observation is built.
        for sim in env.agents.values():
            self.set_target_speed_mps(sim, self.get_current_speed_mps(sim))
        return super().reset(env)

    def get_obs(self, env, agent_id):
        norm_obs = super().get_obs(env, agent_id)
        norm_obs[2] = (
            self.get_speed_error_mps(env.agents[agent_id])
            / self.speed_error_scale_mps
        )
        return np.clip(
            norm_obs,
            self.observation_space.low,
            self.observation_space.high,
        ).astype(np.float32, copy=False)
