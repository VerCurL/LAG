import math

import numpy as np
from wandb import agent
from ..reward_function_base import BaseRewardFunction
from ...utils.utils import get_AO_TA_R

class TeamPostureReward(BaseRewardFunction):
    def __init__(self, config):
        super().__init__(config)
        self.min_dist = getattr(self.config, f'{self.__class__.__name__}_min_dist', 0.5)        # 单位：km
        self.max_dist = getattr(self.config, f'{self.__class__.__name__}_max_dist', 20)         # 单位：km
        self.attack_angle = getattr(self.config, f'EnmPostureReward_max_attack_angle', 45)      # 单位：°

        self.attack_angle = math.radians(self.attack_angle)

    def get_reward(self, task, env, agent_id):
        new_reward = 0
        PAOs = []
        ego_feature = np.hstack([env.agents[agent_id].get_position(),
                                 env.agents[agent_id].get_velocity()])

        for partner in env.agents[agent_id].partners:
            partner_feature = np.hstack([partner.get_position(),
                                         partner.get_velocity()])
            _, _, R = get_AO_TA_R(ego_feature, partner_feature)
            # print("TeamPostureReward: partner_R = {}".format(R))
            for enm in partner.enemies:
                enm_feature = np.hstack([enm.get_position(),
                                         enm.get_velocity()])
                PAO, _, _ = get_AO_TA_R(partner_feature, enm_feature)
                PAOs.append(PAO)

        # new_reward += self.get_dist_function(R)
        new_reward += self.get_partner_function(R, PAOs)

        # print("TeamPostureReward: partner_reward = {}".format(new_reward))

        return self._process(new_reward, agent_id)

    def get_dist_function(self, R):
        if R > self.max_dist * 1000:
            return -3
        elif R < self.min_dist * 1000:
            return -100
        return 0

    def in_partner_dist(self, R):
        if R < self.max_dist and R > self.min_dist:
            return True
        return False

    def get_partner_function(self, R, PAOs):
        # if self.in_partner_dist(R):
        if abs(PAOs[0]) < self.attack_angle and abs(PAOs[1]) < self.attack_angle:
            return 7
        elif abs(PAOs[0]) < self.attack_angle or abs(PAOs[1]) < self.attack_angle:
            return 3
        return 0