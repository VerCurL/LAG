"""C27 isolated command governance over frozen C23-best and unchanged actuators."""

import math

from .c23_hybrid_hierarchical_singlecombat_task import C23HybridHierarchicalSingleCombatTask
from ..core.catalog import Catalog as c
from ..utils.maneuver_governor import GovernorConfig, govern_maneuver


LOW_LEVEL_SHA256 = "ea1012e201481e315773abdcdbe7d0be3ac39b83b640753f9ba4f74170ecb507"


def flight_state(sim):
    u, v, w = sim.get_property_values([c.velocities_u_mps, c.velocities_v_mps, c.velocities_w_mps])
    return {
        "altitude_m": float(sim.get_property_value(c.position_h_sl_m)),
        "speed_mps": math.sqrt(u * u + v * v + w * w),
        "calibrated_speed_mps": float(sim.get_property_value(c.velocities_vc_mps)),
        "vertical_speed_mps": float(sim.get_property_value(c.velocities_h_dot_fps)) * 0.3048,
        "roll_deg": float(sim.get_property_value(c.attitude_phi_deg)),
        "alpha_deg": float(sim.get_property_value(c.aero_alpha_deg)),
        "beta_deg": float(sim.get_property_value(c.aero_beta_deg)),
        "nx_g": float(sim.get_property_value(c.accelerations_n_pilot_x_norm)),
        "ny_g": float(sim.get_property_value(c.accelerations_n_pilot_y_norm)),
        "nz_g": float(sim.get_property_value(c.accelerations_n_pilot_z_norm)),
    }


class C27GovernedHybridTask(C23HybridHierarchicalSingleCombatTask):
    def __init__(self, config):
        self.governor_config = GovernorConfig(**dict(getattr(config, "C27Governor_parameters", {})))
        self.governor_enabled = getattr(config, "C27Governor_enable", None)
        if not isinstance(self.governor_enabled, bool):
            raise ValueError("C27Governor_enable must be an explicit boolean")
        if getattr(config, "C23LowLevel_actor_sha256", "").lower() != LOW_LEVEL_SHA256:
            raise ValueError("C27 requires the frozen C23 actor_best identity")
        if not str(getattr(config, "C23LowLevel_actor_path", "")).endswith("/actor_best.pt"):
            raise ValueError("C27 requires actor_best.pt")
        if getattr(config, "sim_freq", None) != 60 or getattr(config, "agent_interaction_steps", None) != 12:
            raise ValueError("C27 requires 5 Hz control / 60 Hz simulation")
        super().__init__(config)
        self._c27_substeps = {}

    def reset(self, env):
        result = super().reset(env)
        self._c27_substeps = {agent_id: [] for agent_id in env.ego_ids}
        return result

    def _govern_maneuver_offsets(self, sim, altitude_m, heading_deg, speed_mps):
        state = flight_state(sim)
        return govern_maneuver(
            altitude_m, heading_deg, speed_mps,
            altitude_m=state["altitude_m"], speed_mps=state["speed_mps"],
            vertical_speed_mps=state["vertical_speed_mps"],
            enabled=self.governor_enabled, config=self.governor_config,
        )

    def normalize_action(self, env, agent_id, action):
        if agent_id in env.ego_ids:
            self._c27_substeps[agent_id] = []
        return super().normalize_action(env, agent_id, action)

    def sample_substep_action(self, env, agent_id, substep_index):
        # Before substep k>0 we observe the state after k-1; step() captures
        # the last substep. Exactly 12 distinct post-integration samples.
        if agent_id in env.ego_ids and substep_index > 0:
            self._c27_substeps[agent_id].append(flight_state(env.agents[agent_id]))
        return super().sample_substep_action(env, agent_id, substep_index)

    def step(self, env):
        for agent_id in env.ego_ids:
            self._c27_substeps[agent_id].append(flight_state(env.agents[agent_id]))
        return super().step(env)

    def get_substep_states(self, agent_id):
        return [dict(row) for row in self._c27_substeps[agent_id]]
