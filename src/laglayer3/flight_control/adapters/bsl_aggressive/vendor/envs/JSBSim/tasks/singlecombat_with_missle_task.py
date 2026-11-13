import numpy as np
from gymnasium import spaces
from collections import deque

from .singlecombat_task import SingleCombatTask, HierarchicalSingleCombatTask
from ..reward_functions import AltitudeReward, PostureReward, MissilePostureReward, EventDrivenReward, ShootPenaltyReward, ControlSmoothnessReward, MissileRightEvasionReward, AntiLoiterReward
from ..core.catalog import Catalog as c
from ..core.simulatior import MissileSimulator
from ..decision import TacticalPipeline
from ..utils.utils import LLA2NEU, get_AO_TA_R


class SingleCombatDodgeMissileTask(SingleCombatTask):
    """This task aims at training agent to dodge missile attacking
    """
    def __init__(self, config):
        super().__init__(config)

        self.max_attack_angle = getattr(self.config, 'max_attack_angle', 180)
        self.max_attack_distance = getattr(self.config, 'max_attack_distance', np.inf)
        self.min_attack_interval = getattr(self.config, 'min_attack_interval', 125)
        # Optional scripted evasion: once enemy launches, force a right maneuver for K decisions.
        self.evade_right_enable = getattr(self.config, 'EvadeRightOnMissile_enable', False)
        self.evade_right_steps = getattr(self.config, 'EvadeRightOnMissile_steps', 0)
        self.evade_right_aileron = getattr(self.config, 'EvadeRightOnMissile_aileron', 1.0)
        self.evade_right_elevator = getattr(self.config, 'EvadeRightOnMissile_elevator', 0.0)
        self.evade_right_rudder = getattr(self.config, 'EvadeRightOnMissile_rudder', 1.0)
        self.evade_right_throttle = getattr(self.config, 'EvadeRightOnMissile_throttle', 0.9)
        self.reward_functions = [
            PostureReward(self.config),
            MissilePostureReward(self.config),
            AltitudeReward(self.config),
            ControlSmoothnessReward(self.config),
            MissileRightEvasionReward(self.config),
            AntiLoiterReward(self.config),
            EventDrivenReward(self.config)
        ]
        self.decision_enable = getattr(self.config, "Decision_enable", False)
        self._decision_ctx = {}
        self._tactical_pipelines = {}

    def load_observation_space(self):
        self.observation_space = spaces.Box(low=-10, high=10., shape=(21,))

    def get_obs(self, env, agent_id):
        """
        Convert simulation states into the format of observation_space

        ------
        Returns: (np.ndarray)
        - ego info
            - [0] ego altitude           (unit: 5km)
            - [1] ego_roll_sin
            - [2] ego_roll_cos
            - [3] ego_pitch_sin
            - [4] ego_pitch_cos
            - [5] ego v_body_x           (unit: mh)
            - [6] ego v_body_y           (unit: mh)
            - [7] ego v_body_z           (unit: mh)
            - [8] ego_vc                 (unit: mh)
        - relative enm info
            - [9] delta_v_body_x         (unit: mh)
            - [10] delta_altitude        (unit: km)
            - [11] ego_AO                (unit: rad) [0, pi]
            - [12] ego_TA                (unit: rad) [0, pi]
            - [13] relative distance     (unit: 10km)
            - [14] side_flag             1 or 0 or -1
        - relative missile info
            - [15] delta_v_body_x
            - [16] delta altitude
            - [17] ego_AO
            - [18] ego_TA
            - [19] relative distance
            - [20] side flag
        """
        norm_obs = np.zeros(21)
        ego_obs_list = np.array(env.agents[agent_id].get_property_values(self.state_var))
        enm_obs_list = np.array(env.agents[agent_id].enemies[0].get_property_values(self.state_var))
        # (0) extract feature: [north(km), east(km), down(km), v_n(mh), v_e(mh), v_d(mh)]
        ego_cur_ned = LLA2NEU(*ego_obs_list[:3], env.center_lon, env.center_lat, env.center_alt)
        enm_cur_ned = LLA2NEU(*enm_obs_list[:3], env.center_lon, env.center_lat, env.center_alt)
        ego_feature = np.array([*ego_cur_ned, *ego_obs_list[6:9]])
        enm_feature = np.array([*enm_cur_ned, *enm_obs_list[6:9]])
        # (1) ego info normalization
        norm_obs[0] = ego_obs_list[2] / 5000
        norm_obs[1] = np.sin(ego_obs_list[3])
        norm_obs[2] = np.cos(ego_obs_list[3])
        norm_obs[3] = np.sin(ego_obs_list[4])
        norm_obs[4] = np.cos(ego_obs_list[4])
        norm_obs[5] = ego_obs_list[9] / 340
        norm_obs[6] = ego_obs_list[10] / 340
        norm_obs[7] = ego_obs_list[11] / 340
        norm_obs[8] = ego_obs_list[12] / 340
        # (2) relative enm info
        ego_AO, ego_TA, R, side_flag = get_AO_TA_R(ego_feature, enm_feature, return_side=True)
        norm_obs[9] = (enm_obs_list[9] - ego_obs_list[9]) / 340
        norm_obs[10] = (enm_obs_list[2] - ego_obs_list[2]) / 1000
        norm_obs[11] = ego_AO
        norm_obs[12] = ego_TA
        norm_obs[13] = R / 10000
        norm_obs[14] = side_flag
        # (3) relative missile info
        missile_sim = env.agents[agent_id].check_missile_warning()
        if missile_sim is not None:
            missile_feature = np.concatenate((missile_sim.get_position(), missile_sim.get_velocity()))
            ego_AO, ego_TA, R, side_flag = get_AO_TA_R(ego_feature, missile_feature, return_side=True)
            norm_obs[15] = (np.linalg.norm(missile_sim.get_velocity()) - ego_obs_list[9]) / 340
            norm_obs[16] = (missile_feature[2] - ego_obs_list[2]) / 1000
            norm_obs[17] = ego_AO
            norm_obs[18] = ego_TA
            norm_obs[19] = R / 10000
            norm_obs[20] = side_flag
        return norm_obs

    def _compute_delta_es(self, ego, enm):
        g = 9.80665
        ego_h = float(ego.get_position()[-1])
        enm_h = float(enm.get_position()[-1])
        ego_v = float(np.linalg.norm(ego.get_velocity()))
        enm_v = float(np.linalg.norm(enm.get_velocity()))
        return (ego_h + ego_v * ego_v / (2.0 * g)) - (enm_h + enm_v * enm_v / (2.0 * g))

    def _compute_safety_margin(self, sim):
        h = float(sim.get_position()[-1])
        nz = abs(float(sim.get_property_value(c.accelerations_n_pilot_z_norm)))
        alpha = abs(float(sim.get_property_value(c.aero_alpha_deg)))

        h_min = float(getattr(self.config, "h_min", 2800.0))
        h_soft_min = float(getattr(self.config, "h_soft_min", 3200.0))
        h_soft_max = float(getattr(self.config, "h_soft_max", 9800.0))
        h_max = float(getattr(self.config, "h_max", 10500.0))
        nz_max = float(getattr(self.config, "n_z_max", 8.8))
        alpha_max = float(getattr(self.config, "alpha_max_deg", 24.0))

        m_low_h = (h - h_min) / max(1.0, h_soft_min - h_min)
        m_high_h = (h_max - h) / max(1.0, h_max - h_soft_max)
        m_nz = (nz_max - nz) / max(1e-6, nz_max)
        m_alpha = (alpha_max - alpha) / max(1e-6, alpha_max)
        return float(np.clip(min(m_low_h, m_high_h, m_nz, m_alpha), -1.0, 1.0))

    def get_decision_ctx(self, agent_id):
        return self._decision_ctx.get(agent_id, {})

    def _trigger_evade_right(self, agent_id):
        if self.evade_right_enable and self.evade_right_steps > 0:
            self._forced_right_steps[agent_id] = max(self._forced_right_steps.get(agent_id, 0), int(self.evade_right_steps))

    def _apply_evade_right_action(self, env, agent_id, norm_act):
        if not self.evade_right_enable:
            return norm_act
        if self._forced_right_steps.get(agent_id, 0) <= 0:
            return norm_act
        if not env.agents[agent_id].is_alive:
            return norm_act

        self._forced_right_steps[agent_id] -= 1
        forced = np.array([
            self.evade_right_aileron,
            self.evade_right_elevator,
            self.evade_right_rudder,
            self.evade_right_throttle,
        ], dtype=np.float32)
        forced[:3] = np.clip(forced[:3], -1.0, 1.0)
        forced[3] = np.clip(forced[3], 0.0, 0.9)
        return forced

    def normalize_action(self, env, agent_id, action):
        norm_act = super().normalize_action(env, agent_id, action)
        return self._apply_evade_right_action(env, agent_id, norm_act)

    def reset(self, env):
        """Reset fighter blood & missile status
        """
        self._last_shoot_time = {agent_id: -self.min_attack_interval for agent_id in env.agents.keys()}
        self.remaining_missiles = {agent_id: agent.num_missiles for agent_id, agent in env.agents.items()}
        self.lock_duration = {agent_id: deque(maxlen=int(1 / env.time_interval)) for agent_id in env.agents.keys()}
        self._forced_right_steps = {agent_id: 0 for agent_id in env.agents.keys()}
        self._decision_ctx = {agent_id: {} for agent_id in env.agents.keys()}
        if self.decision_enable:
            self._tactical_pipelines = {
                agent_id: TacticalPipeline(self.config, dt=env.time_interval)
                for agent_id in env.agents.keys()
            }
            for p in self._tactical_pipelines.values():
                p.reset()
        else:
            self._tactical_pipelines = {}
        return super().reset(env)

    def step(self, env):
        SingleCombatTask.step(self, env)
        for agent_id, agent in env.agents.items():
            if self.decision_enable:
                enm = agent.enemies[0]
                missile_sim = agent.check_missile_warning()
                missile_alert = missile_sim is not None

                ego_pos = agent.get_position()
                ego_vel = agent.get_velocity()
                enm_pos = enm.get_position()
                enm_vel = enm.get_velocity()

                missile_pos, missile_vel = None, None
                if missile_alert:
                    missile_pos = missile_sim.get_position()
                    missile_vel = missile_sim.get_velocity()

                ego_feature = np.hstack([ego_pos, ego_vel])
                enm_feature = np.hstack([enm_pos, enm_vel])
                ao_rad, _, _ = get_AO_TA_R(ego_feature, enm_feature)
                ao_deg = float(np.rad2deg(ao_rad))

                delta_es = self._compute_delta_es(agent, enm)
                safety_margin = self._compute_safety_margin(agent)

                ctx = self._tactical_pipelines[agent_id].update(
                    step_idx=env.current_step,
                    ego_pos=ego_pos,
                    ego_vel=ego_vel,
                    enemy_pos=enm_pos,
                    enemy_vel=enm_vel,
                    ao_deg=ao_deg,
                    delta_es=delta_es,
                    safety_margin=safety_margin,
                    missile_alert=missile_alert,
                    missile_pos=missile_pos,
                    missile_vel=missile_vel,
                )
                self._decision_ctx[agent_id] = ctx
            else:
                self._decision_ctx[agent_id] = {}

            # [Rule-based missile launch]
            target = agent.enemies[0].get_position() - agent.get_position()
            heading = agent.get_velocity()
            distance = np.linalg.norm(target)
            attack_angle = np.rad2deg(np.arccos(np.clip(np.sum(target * heading) / (distance * np.linalg.norm(heading) + 1e-8), -1, 1)))
            self.lock_duration[agent_id].append(attack_angle < self.max_attack_angle)
            shoot_interval = env.current_step - self._last_shoot_time[agent_id]

            shoot_flag = agent.is_alive and np.sum(self.lock_duration[agent_id]) >= self.lock_duration[agent_id].maxlen \
                and distance <= self.max_attack_distance and self.remaining_missiles[agent_id] > 0 and shoot_interval >= self.min_attack_interval
            if shoot_flag:
                new_missile_uid = agent_id + str(self.remaining_missiles[agent_id])
                env.add_temp_simulator(
                    MissileSimulator.create(parent=agent, target=agent.enemies[0], uid=new_missile_uid))
                self.remaining_missiles[agent_id] -= 1
                self._last_shoot_time[agent_id] = env.current_step
                self._trigger_evade_right(agent.enemies[0].uid)


class HierarchicalSingleCombatDodgeMissileTask(HierarchicalSingleCombatTask, SingleCombatDodgeMissileTask):

    def __init__(self, config: str):
        HierarchicalSingleCombatTask.__init__(self, config)

        self.reward_functions = [
            PostureReward(self.config),
            MissilePostureReward(self.config),
            AltitudeReward(self.config),
            ControlSmoothnessReward(self.config),
            MissileRightEvasionReward(self.config),
            AntiLoiterReward(self.config),
            EventDrivenReward(self.config)
        ]

    def load_observation_space(self):
        return SingleCombatDodgeMissileTask.load_observation_space(self)

    def load_action_space(self):
        return HierarchicalSingleCombatTask.load_action_space(self)

    def get_obs(self, env, agent_id):
        return SingleCombatDodgeMissileTask.get_obs(self, env, agent_id)

    def normalize_action(self, env, agent_id, action):
        norm_act = HierarchicalSingleCombatTask.normalize_action(self, env, agent_id, action)
        return self._apply_evade_right_action(env, agent_id, norm_act)

    def reset(self, env):
        self._inner_rnn_states = {agent_id: np.zeros((1, 1, 128)) for agent_id in env.agents.keys()}
        return SingleCombatDodgeMissileTask.reset(self, env)

    def step(self, env):
        return SingleCombatDodgeMissileTask.step(self, env)


class SingleCombatShootMissileTask(SingleCombatDodgeMissileTask):
    def __init__(self, config):
        super().__init__(config)

        self.reward_functions = [
            PostureReward(self.config),
            AltitudeReward(self.config),
            ControlSmoothnessReward(self.config),
            EventDrivenReward(self.config),
            ShootPenaltyReward(self.config)
        ]

    def load_observation_space(self):
        self.observation_space = spaces.Box(low=-10, high=10., shape=(21,))

    def load_action_space(self):
        # aileron, elevator, rudder, throttle, shoot control
        self.action_space = spaces.Tuple([spaces.MultiDiscrete([41, 41, 41, 30]), spaces.Discrete(2)])
    
    def get_obs(self, env, agent_id):
        return super().get_obs(env, agent_id)
    
    def normalize_action(self, env, agent_id, action):
        self._shoot_action[agent_id] = action[-1]
        return super().normalize_action(env, agent_id, action[:-1].astype(np.int32))
    
    def reset(self, env):
        self._shoot_action = {agent_id: 0 for agent_id in env.agents.keys()}
        self.remaining_missiles = {agent_id: agent.num_missiles for agent_id, agent in env.agents.items()}
        super().reset(env)
    
    def step(self, env):
        SingleCombatTask.step(self, env)
        for agent_id, agent in env.agents.items():
            # [RL-based missile launch with limited condition]
            shoot_flag = agent.is_alive and self._shoot_action[agent_id] and self.remaining_missiles[agent_id] > 0
            if shoot_flag:
                new_missile_uid = agent_id + str(self.remaining_missiles[agent_id])
                env.add_temp_simulator(
                    MissileSimulator.create(parent=agent, target=agent.enemies[0], uid=new_missile_uid))
                self.remaining_missiles[agent_id] -= 1
                self._trigger_evade_right(agent.enemies[0].uid)


class HierarchicalSingleCombatShootTask(HierarchicalSingleCombatTask, SingleCombatShootMissileTask):
    def __init__(self, config: str):
        HierarchicalSingleCombatTask.__init__(self, config)
        self.reward_functions = [
            PostureReward(self.config),
            AltitudeReward(self.config),
            ControlSmoothnessReward(self.config),
            EventDrivenReward(self.config),
            ShootPenaltyReward(self.config)
        ]

    def load_observation_space(self):
        return SingleCombatShootMissileTask.load_observation_space(self)

    def load_action_space(self):
        # altitude control + heading control + velocity control + shoot control
        self.action_space = spaces.Tuple([spaces.MultiDiscrete([3, 5, 3]), spaces.Discrete(2)])

    def get_obs(self, env, agent_id):
        return SingleCombatShootMissileTask.get_obs(self, env, agent_id)

    def normalize_action(self, env, agent_id, action):
        """Convert high-level action into low-level action.
        """
        self._shoot_action[agent_id] = action[-1]
        norm_act = HierarchicalSingleCombatTask.normalize_action(self, env, agent_id, action[:-1].astype(np.int32))
        return self._apply_evade_right_action(env, agent_id, norm_act)

    def reset(self, env):
        self._inner_rnn_states = {agent_id: np.zeros((1, 1, 128)) for agent_id in env.agents.keys()}
        SingleCombatShootMissileTask.reset(self, env)

    def step(self, env):
        SingleCombatShootMissileTask.step(self, env)
