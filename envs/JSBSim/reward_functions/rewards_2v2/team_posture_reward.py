import numpy as np
from wandb import agent
from ..reward_function_base import BaseRewardFunction
from ...utils.utils import get_AO_TA_R

class TeamPostureReward(BaseRewardFunction):
    def __init__(self, config):
        super().__init__(config)
        self.min_dist = getattr(self.config, f'{self.__class__.__name__}_min_dist', 5)
        self.max_dist = getattr(self.config, f'{self.__class__.__name__}_max_dist', 40)
        self.attack_angle = getattr(self.config, f'EnmPostureReward_max_attack_angle', 45)

    def get_reward(self, task, env, agent_id):
        new_reward = 0
        PAOs = []
        ego_feature = np.hstack([env.agents[agent_id].get_position(),
                                 env.agents[agent_id].get_velocity()])

        for partner in env.agents[agent_id].partners:
            partner_feature = np.hstack([partner.get_position(),
                                         partner.get_velocity()])
            _, _, R = get_AO_TA_R(ego_feature, partner_feature)
            for enm in partner.enemies:
                enm_feature = np.hstack([enm.get_position(),
                                         enm.get_velocity()])
                PAO, _, _ = get_AO_TA_R(partner_feature, enm_feature)
                PAOs.append(PAO)

        new_reward += self.get_dist_function(R)
        new_reward += self.get_partner_function(R, PAOs)

        return self._process(new_reward, agent_id)

    def get_dist_function(self, R):
        if R > self.max_dist:
            return -3
        elif R < self.min_dist:
            return -100

    def in_attack_angle(self, AO):
        if AO > -self.attack_angle and AO < self.attack_angle:
            return True
        return False

    def in_partner_dist(self, R):
        if R < self.max_dist and R > self.min_dist:
            return True
        return False

    def get_partner_function(self, R, PAOs):
        if self.in_partner_dist(R):
            if self.in_attack_angle(PAOs[0]) and self.in_attack_angle(PAOs[1]):
                return 7
            elif self.in_attack_angle(PAOs[0]) or self.in_attack_angle(PAOs[1]):
                return 3
        return 0