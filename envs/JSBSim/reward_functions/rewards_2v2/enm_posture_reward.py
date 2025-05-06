import numpy as np
import math
from wandb import agent
from envs.JSBSim.reward_functions.reward_function_base import BaseRewardFunction
from envs.JSBSim.utils.utils import get_AO_TA_R

class EnmPostureReward(BaseRewardFunction):
    def __init__(self, config):
        super().__init__(config)
        self.attack_angle = getattr(self.config, f'{self.__class__.__name__}_max_attack_angle', 45)
        self.target_dist = getattr(self.config, f'{self.__class__.__name__}_target_dist', 30)

        self.attack_angle = math.radians(self.attack_angle)

    def get_reward(self, task, env, agent_id):
        new_reward = 0
        egoAOs = []
        enmAOs = []
        Rs = []

        ego_feature = np.hstack([env.agents[agent_id].get_position(),
                                 env.agents[agent_id].get_velocity()])

        for enm in env.agents[agent_id].enemies:
            enm_feature = np.hstack([enm.get_position(),
                                     enm.get_velocity()])
            AO, TA, R = get_AO_TA_R(ego_feature, enm_feature)
            egoAOs.append(AO)
            enmAOs.append(math.radians(180) - TA)
            Rs.append(R)

        # print("EnmPostureReward: ego_reward = {}, enm_reward = {}".format(self.get_env_attack_angle_function(egoAOs), self.get_env_attack_angle_function(enmAOs)))
        new_reward += self.get_env_attack_angle_function(egoAOs)
        new_reward -= self.get_env_attack_angle_function(enmAOs)

        return self._process(new_reward, agent_id)


    def get_env_attack_angle_function(self, AOs):
        if abs(AOs[0]) < self.attack_angle and abs(AOs[1]) < self.attack_angle:
            return 10
        elif abs(AOs[0]) < self.attack_angle or abs(AOs[1]) < self.attack_angle:
            return 5
        return 0


    def in_dist(self, R):
        if R < self.target_dist:
            return True
        return False

    def get_target_dist_function(self, Rs):
        if self.in_dist(Rs[0]) and self.in_dist(Rs[1]):
            return 3
        elif self.in_dist(Rs[0]) or self.in_dist(Rs[1]):
            return 1
        return -10
