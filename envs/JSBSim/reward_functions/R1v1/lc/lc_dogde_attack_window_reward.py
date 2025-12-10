import numpy as np
from .reward_function_base import BaseRewardFunction
import math
from envs.JSBSim.core.catalog import Catalog as c
from envs.JSBSim.utils.utils import LLA2NEU, get_AO_TA_R


class DogdeAttackReward(BaseRewardFunction):
    def __init__(self, config):
        super().__init__(config)
        self.R_max_Angle_threshold = math.radians(60)
        self.AO_threshold = math.radians(90)
        self.t1 = 2
        self.t2 = 50

        self.previous_TA = {}

    def get_reward(self, task, env, agent_id):

        ego_obs_list = np.array(env._jsbsims[agent_id].get_property_values(task.state_var))
        enm_obs_list = np.array(env._jsbsims[agent_id].enemies[0].get_property_values(task.state_var))
        ego_cur_ned = LLA2NEU(*ego_obs_list[:3], 123.4, 26.0, 0.0)
        enm_cur_ned = LLA2NEU(*enm_obs_list[:3], 123.4, 26.0, 0.0)
        ego_feature = np.array([*ego_cur_ned, *(ego_obs_list[6:9])])
        enm_feature = np.array([*enm_cur_ned, *(enm_obs_list[6:9])])
        AO, TA, R_dis, side_flage = get_AO_TA_R(ego_feature, enm_feature, return_side=True)
        R_ego_min = task.min_missile_attack_distance
        R_ego_max = task.max_missile_attack_distance

        enm_AO = math.pi - TA
        if agent_id in self.previous_TA:
            enm_AO_previous = math.pi - self.previous_TA[agent_id]
            dealt_angel = enm_AO_previous - enm_AO

        self.previous_TA[agent_id] = TA

        remaining = 1 if task.remaining_missiles[agent_id] > 0 else 0
        reward = 0
        if remaining:
            if R_ego_min * 0.9 <= R_dis <= 1.3 * R_ego_max:
                if abs(enm_AO) < self.R_max_Angle_threshold:
                    reward = -self.t1 * abs(2-enm_AO) if dealt_angel > 0 else -self.t2 * dealt_angel
                elif abs(enm_AO) > self.R_max_Angle_threshold:
                    reward = abs(enm_AO) - 1 if dealt_angel > 0 else 0.5
        else:
            reward = abs(enm_AO - 1) - self.t2 * dealt_angel
        return reward
