import numpy as np
from .reward_function_base import BaseRewardFunction
import math
from envs.JSBSim.core.catalog import Catalog as c
from envs.JSBSim.utils.utils import LLA2NEU, get_AO_TA_R

class AttackWindowReward(BaseRewardFunction):
    def __init__(self, config):
        super().__init__(config)
        self.R_max_Angle_threshold = math.radians(60)
        self.AO_threshold = math.radians(90)
        self.a1 = 100
        self.a2 = 0.2
        self.y1 = 0.01
        self.d1 = 0.003
        self.d2 = 0.03
        self.d3 = 0.1
        self.yaw_threshold = math.radians(30)
        self.yaw_weight = 0.5

        self.previous_status = {}

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

        # 上一步的AO和R
        dealt_angel = 0
        dealt_d = 0
        if agent_id in self.previous_status:
            # 后面计算AO和R的变化量
            dealt_angel = self.previous_status[agent_id]['AO'] - AO
            dealt_d = self.previous_status[agent_id]['R_dis'] - R_dis
        else:
            self.previous_status[agent_id] = {}

        # 缓存当前时刻的值
        self.previous_status[agent_id]['AO'] = AO
        self.previous_status[agent_id]['R_dis'] = R_dis

        remaining = 1 if task.remaining_missiles[agent_id] > 0 else 0

        # 避免变化过大
        reward = 0
        if abs(dealt_d) > 300:
            dealt_d = 0
        if abs(dealt_angel) > 0.1:
            dealt_angel = 0

        if remaining:
            if R_ego_min * 1.1 <= R_dis <= 1.1 * R_ego_max:
                if abs(AO) < self.R_max_Angle_threshold:
                    reward = self.d1 * dealt_d + self.a1 * dealt_angel
                elif abs(AO) > self.R_max_Angle_threshold:
                    reward = self.y1 * dealt_d - self.a2 * abs(AO)
            elif R_dis > 1.1 * R_ego_max:
                reward = self.d2 * dealt_d + (1.57-AO)
            else:
                reward = -self.d2 * dealt_d
        else:
            reward = - self.d3 * dealt_d

        return reward
