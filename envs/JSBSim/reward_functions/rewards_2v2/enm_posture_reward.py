import numpy as np
from wandb import agent
from envs.JSBSim.reward_functions.reward_function_base import BaseRewardFunction
from envs.JSBSim.utils.utils import get_AO_TA_R

class EnmPostureReward(BaseRewardFunction):
    def __init__(self, config):
        super().__init__(config)
        self.attack_angle = getattr(self.config, f'{self.__class__.__name__}_max_attack_angle', 45)
        self.target_dist = getattr(self.config, f'{self.__class__.__name__}_target_dist', 30)

    def get_reward(self, task, env, agent_id):
        new_reward = 0
        AOs = []
        TAs = []
        Rs = []

        ego_feature = np.hstack([env.agents[agent_id].get_position(),
                                 env.agents[agent_id].get_velocity()])

        for enm in env.agents[agent_id].enemies:
            enm_feature = np.hstack([enm.get_position(),
                                     enm.get_velocity()])
            AO, TA, R = get_AO_TA_R(ego_feature, enm_feature)
            AOs.append(AO)
            TAs.append(TA)
            Rs.append(R)

        new_reward += self.get_env_attack_angle_function(AOs)
        new_reward += self.get_enm_attack_angle_function(TAs)

        return self._process(new_reward, agent_id)

    def in_attack_angle(self, AO):
        if AO > -self.attack_angle and AO < self.attack_angle:
            return True
        return False

    def get_env_attack_angle_function(self, AOs):
        if self.in_attack_angle(AOs[0]) and self.in_attack_angle(AOs[1]):
            return 10
        elif self.in_attack_angle(AOs[0]) or self.in_attack_angle(AOs[1]):
            return 5
        return 0

    def get_enm_attack_angle_function(self, TAs):
        if self.in_attack_angle(TAs[0]) and self.in_attack_angle(TAs[1]):
            return -10
        elif self.in_attack_angle(TAs[0]) or self.in_attack_angle(TAs[1]):
            return -5
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
