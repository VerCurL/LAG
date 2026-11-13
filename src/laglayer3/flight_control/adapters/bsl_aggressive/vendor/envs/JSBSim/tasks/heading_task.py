import numpy as np
from gymnasium import spaces
from .task_base import BaseTask
from ..core.catalog import Catalog as c
from ..reward_functions import AltitudeReward, AttitudeSmoothnessReward, ControlSmoothnessReward, ElevatorSecondDiffReward, HeadingReward, LongitudinalDampingReward, LongitudinalMIMOStabilityReward, PowerMomentumReward, QPAlignReward, TargetDeltaSmoothnessReward, TrimBalanceReward
from ..termination_conditions import ExtremeState, LowAltitude, Overload, Timeout, UnreachHeading
from ..utils.target_rate_observation import compute_route_target_rates


class HeadingTask(BaseTask):
    '''
    Control target heading.
    '''
    def __init__(self, config):
        super().__init__(config)

        self.reward_functions = [
            HeadingReward(self.config),
            PowerMomentumReward(self.config),
            AltitudeReward(self.config),
            AttitudeSmoothnessReward(self.config),
            ControlSmoothnessReward(self.config),
            QPAlignReward(self.config),
            LongitudinalDampingReward(self.config),
            ElevatorSecondDiffReward(self.config),
            TargetDeltaSmoothnessReward(self.config),
            LongitudinalMIMOStabilityReward(self.config),
            TrimBalanceReward(self.config),
        ]
        self.termination_conditions = [
            UnreachHeading(self.config),
            ExtremeState(self.config),
            Overload(self.config),
            LowAltitude(self.config),
            Timeout(self.config),
        ]

    @property
    def num_agents(self):
        return 1

    def load_variables(self):
        self.action_type = getattr(self.config, "action_type", "discrete")
        self.obs_version = getattr(self.config, "obs_version", "v1")
        self.state_var = [
            c.delta_altitude,                   # 0. delta_h   (unit: m)
            c.delta_heading,                    # 1. delta_heading  (unit: °)
            c.delta_velocities_u,               # 2. delta_v   (unit: m/s)
            c.position_h_sl_m,                  # 3. altitude  (unit: m)
            c.attitude_roll_rad,                # 4. roll      (unit: rad)
            c.attitude_pitch_rad,               # 5. pitch     (unit: rad)
            c.velocities_u_mps,                 # 6. v_body_x   (unit: m/s)
            c.velocities_v_mps,                 # 7. v_body_y   (unit: m/s)
            c.velocities_w_mps,                 # 8. v_body_z   (unit: m/s)
            c.velocities_vc_mps,                # 9. vc        (unit: m/s)
        ]
        if self.obs_version == "v2_rates":
            self.state_var.extend([
                c.velocities_p_rad_sec,         # 10. roll rate  (unit: rad/s)
                c.velocities_q_rad_sec,         # 11. pitch rate (unit: rad/s)
                c.velocities_r_rad_sec,         # 12. yaw rate   (unit: rad/s)
            ])
        elif self.obs_version in (
            "v3_core",
            "v4_qp_state",
            "v5_target_rates",
        ):
            self.state_var.extend([
                c.velocities_p_rad_sec,         # 10. roll rate  (unit: rad/s)
                c.velocities_q_rad_sec,         # 11. pitch rate (unit: rad/s)
                c.velocities_r_rad_sec,         # 12. yaw rate   (unit: rad/s)
                c.aero_alpha_deg,               # 13. AOA alpha  (unit: deg)
                c.velocities_h_dot_fps,         # 14. altitude rate (unit: ft/s)
                c.fcs_aileron_cmd_norm,         # 15. previous aileron command
                c.fcs_elevator_cmd_norm,        # 16. previous elevator command
                c.fcs_rudder_cmd_norm,          # 17. previous rudder command
                c.fcs_throttle_cmd_norm,        # 18. previous throttle command [0.4, 0.9]
            ])
        self.action_var = [
            c.fcs_aileron_cmd_norm,             # [-1., 1.]
            c.fcs_elevator_cmd_norm,            # [-1., 1.]
            c.fcs_rudder_cmd_norm,              # [-1., 1.]
            c.fcs_throttle_cmd_norm,            # [0.4, 0.9]
        ]
        self.render_var = [
            c.position_long_gc_deg,
            c.position_lat_geod_deg,
            c.position_h_sl_m,
            c.attitude_roll_rad,
            c.attitude_pitch_rad,
            c.attitude_heading_true_rad,
        ]

    def load_observation_space(self):
        obs_version = getattr(self, "obs_version", "v1")
        if obs_version == "v5_target_rates":
            obs_dim = 32
            scales = np.asarray(
                [
                    float(
                        getattr(
                            self.config,
                            "TargetRateObservation_heading_scale_deg_s",
                            3.0,
                        )
                    ),
                    float(
                        getattr(
                            self.config,
                            "TargetRateObservation_altitude_scale_mps",
                            10.0,
                        )
                    ),
                    float(
                        getattr(
                            self.config,
                            "TargetRateObservation_speed_scale_mps2",
                            0.2,
                        )
                    ),
                ],
                dtype=np.float32,
            )
            if not np.all(np.isfinite(scales)) or np.any(scales <= 0.0):
                raise ValueError(
                    "target-rate observation scales must be finite and positive"
                )
            self.target_rate_observation_scales = scales
        elif obs_version == "v4_qp_state":
            obs_dim = 29
        elif obs_version == "v3_core":
            obs_dim = 21
        elif obs_version == "v2_rates":
            obs_dim = 15
        else:
            obs_dim = 12
        self.observation_space = spaces.Box(low=-10, high=10., shape=(obs_dim,))

    def load_action_space(self):
        # aileron, elevator, rudder, throttle
        if getattr(self, "action_type", "discrete") == "continuous":
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        else:
            self.action_space = spaces.MultiDiscrete([41, 41, 41, 30])

    def get_obs(self, env, agent_id):
        """
        Convert simulation states into the format of observation_space.

        observation(dim 12):
            0. ego delta altitude      (unit: km)
            1. ego delta heading       (unit rad)
            2. ego delta velocities_u  (unit: mh)
            3. ego_altitude            (unit: 5km)
            4. ego_roll_sin
            5. ego_roll_cos
            6. ego_pitch_sin
            7. ego_pitch_cos
            8. ego v_body_x            (unit: mh)
            9. ego v_body_y            (unit: mh)
            10. ego v_body_z           (unit: mh)
            11. ego_vc                 (unit: mh)
        """
        obs = np.array(env.agents[agent_id].get_property_values(self.state_var))
        norm_obs = np.zeros(self.observation_space.shape[0], dtype=np.float32)
        norm_obs[0] = obs[0] / 1000         # 0. ego delta altitude (unit: 1km)
        norm_obs[1] = obs[1] / 180 * np.pi  # 1. ego delta heading  (unit rad)
        norm_obs[2] = obs[2] / 340          # 2. ego delta velocities_u (unit: mh)
        norm_obs[3] = obs[3] / 5000         # 3. ego_altitude   (unit: 5km)
        norm_obs[4] = np.sin(obs[4])        # 4. ego_roll_sin
        norm_obs[5] = np.cos(obs[4])        # 5. ego_roll_cos
        norm_obs[6] = np.sin(obs[5])        # 6. ego_pitch_sin
        norm_obs[7] = np.cos(obs[5])        # 7. ego_pitch_cos
        norm_obs[8] = obs[6] / 340          # 8. ego_v_north    (unit: mh)
        norm_obs[9] = obs[7] / 340          # 9. ego_v_east     (unit: mh)
        norm_obs[10] = obs[8] / 340         # 10. ego_v_down    (unit: mh)
        norm_obs[11] = obs[9] / 340         # 11. ego_vc        (unit: mh)
        if self.obs_version in (
            "v2_rates",
            "v3_core",
            "v4_qp_state",
            "v5_target_rates",
        ):
            norm_obs[12] = obs[10] / (2 * np.pi)  # 12. roll rate
            norm_obs[13] = obs[11] / (2 * np.pi)  # 13. pitch rate
            norm_obs[14] = obs[12] / (2 * np.pi)  # 14. yaw rate
        if self.obs_version in (
            "v3_core",
            "v4_qp_state",
            "v5_target_rates",
        ):
            h_dot_mps = obs[14] * 0.3048
            norm_obs[15] = obs[13] / 15.0
            norm_obs[16] = h_dot_mps / 10.0
            norm_obs[17] = obs[15]
            norm_obs[18] = obs[16]
            norm_obs[19] = obs[17]
            throttle_min, throttle_max = self._throttle_command_bounds()
            throttle_center = 0.5 * (throttle_min + throttle_max)
            throttle_half_range = 0.5 * (throttle_max - throttle_min)
            norm_obs[20] = (
                obs[18] - throttle_center
            ) / throttle_half_range
        if self.obs_version in ("v4_qp_state", "v5_target_rates"):
            qp_state = np.asarray(env.get_qp_state_observation(agent_id), dtype=np.float32)
            if qp_state.shape != (8,) or not np.all(np.isfinite(qp_state)):
                raise RuntimeError("QP state observation must be a finite 8D vector")
            norm_obs[21:29] = qp_state
        if self.obs_version == "v5_target_rates":
            target_rates = self.get_target_rate_commands(env, agent_id)
            norm_obs[29:32] = (
                target_rates / self.target_rate_observation_scales
            )
        norm_obs = np.clip(norm_obs, self.observation_space.low, self.observation_space.high)
        return norm_obs

    def get_target_rate_commands(self, env, agent_id):
        """Return the target generator's exact mixed-unit rate command."""

        if bool(getattr(self, "_external_target_mode", False)):
            external = getattr(self, "_external_target_rates", None)
            if external is None:
                return np.zeros(3, dtype=np.float32)
            rates = np.asarray(external, dtype=np.float32).reshape(3)
        else:
            sim_time_sec = float(
                env.agents[agent_id].get_property_value(
                    c.simulation_sim_time_sec
                )
            )
            rates = compute_route_target_rates(
                getattr(env, "heading_route_state", None),
                sim_time_sec,
            )
        if rates.shape != (3,) or not np.all(np.isfinite(rates)):
            raise RuntimeError(
                "target-rate command must be a finite 3D vector"
            )
        return rates.astype(np.float32, copy=False)

    def _throttle_command_bounds(self):
        throttle_min = float(
            getattr(self.config, "ActionThrottle_physical_min", 0.4)
        )
        throttle_max = float(
            getattr(self.config, "ActionThrottle_physical_max", 0.9)
        )
        if (
            not np.isfinite(throttle_min)
            or not np.isfinite(throttle_max)
            or throttle_min < 0.0
            or throttle_max > 1.0
            or throttle_min >= throttle_max
        ):
            raise ValueError(
                "physical throttle bounds must satisfy "
                "0 <= min < max <= 1"
            )
        return throttle_min, throttle_max

    def action_to_policy_coordinates(self, action):
        """Convert a four-channel JSBSim command to policy units."""
        policy_action = np.asarray(action, dtype=np.float32).reshape(4).copy()
        throttle_min, throttle_max = self._throttle_command_bounds()
        throttle_center = 0.5 * (throttle_min + throttle_max)
        throttle_half_range = 0.5 * (throttle_max - throttle_min)
        policy_action[3] = (
            policy_action[3] - throttle_center
        ) / throttle_half_range
        return policy_action

    def normalize_action(self, env, agent_id, action):
        """Convert policy action into JSBSim normalized control commands."""
        if self.action_type == "continuous":
            raw = np.asarray(action, dtype=np.float32)
            raw = np.clip(raw, -1.0, 1.0)
            throttle_min, throttle_max = self._throttle_command_bounds()
            throttle_center = 0.5 * (throttle_min + throttle_max)
            throttle_half_range = 0.5 * (throttle_max - throttle_min)
            norm_act = np.zeros(4, dtype=np.float32)
            norm_act[0] = raw[0]
            norm_act[1] = raw[1]
            norm_act[2] = raw[2]
            norm_act[3] = (
                raw[3] * throttle_half_range + throttle_center
            )
            return norm_act

        throttle_min, throttle_max = self._throttle_command_bounds()
        norm_act = np.zeros(4)
        norm_act[0] = action[0] * 2. / (self.action_space.nvec[0] - 1.) - 1.
        norm_act[1] = action[1] * 2. / (self.action_space.nvec[1] - 1.) - 1.
        norm_act[2] = action[2] * 2. / (self.action_space.nvec[2] - 1.) - 1.
        norm_act[3] = (
            action[3]
            * (throttle_max - throttle_min)
            / (self.action_space.nvec[3] - 1.)
            + throttle_min
        )
        return norm_act
