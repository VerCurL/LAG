import math
import numpy as np
from wandb import agent
from envs.JSBSim.reward_functions.reward_function_base import BaseRewardFunction
from envs.JSBSim.utils.utils import get_AO_TA_R

class TeamPostureReward(BaseRewardFunction):
    def __init__(self, config):
        super().__init__(config)
        self.safe_dist_version = getattr(self.config, f'{self.__class__.__name__}_safe_dist_version', 'v0')
        self.help_version = getattr(self.config, f'{self.__class__.__name__}_help_version', 'v0')

        self.min_dist = getattr(self.config, f'{self.__class__.__name__}_min_dist', 0.5)                    # 单位：km
        self.opt_attack_dist = getattr(self.config, f'{self.__class__.__name__}_opt_attack_dist', 12)       # 单位：km
        self.dist_var = getattr(self.config, f'{self.__class__.__name__}_dist_var', 4)

        self.safe_dist_fn = self.safe_dist_function(self.safe_dist_version)
        self.help_fn = self.help_function(self.help_version)

    def get_reward(self, task, env, agent_id):
        new_reward = 0
        ego_feature = np.hstack([env.agents[agent_id].get_position(),
                                 env.agents[agent_id].get_velocity()])

        for partner in env.agents[agent_id].partners:
            partner_feature = np.hstack([partner.get_position(),
                                         partner.get_velocity()])
            _, _, R = get_AO_TA_R(ego_feature, partner_feature)

            safe_dist_reward = self.safe_dist_fn(R / 1000)
            help_reward = self.help_fn(R / 1000)

            new_reward += 5. * safe_dist_reward + 2 * help_reward

        return self._process(new_reward, agent_id)

    def safe_dist_function(self, version):
        """
        和队友保持安全距离的奖励函数，R单位为km
        """
        if version == 'v0':
            return lambda R: -max(0., (self.min_dist - R) / self.min_dist)
        else:
            raise NotImplementedError(f"未知的队友安全距离函数版本: {version}")

    def help_function(self, version):
        """
        队友支援的奖励函数，R单位为km
        """
        if version == 'v0':
            return lambda R: np.exp(-(R - self.opt_attack_dist) ** 2 / self.dist_var)
        else:
            raise NotImplementedError(f"未知的队友支援奖励函数版本: {version}")

