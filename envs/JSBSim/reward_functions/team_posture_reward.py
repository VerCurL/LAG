import math
import numpy as np
from wandb import agent
from .reward_function_base import BaseRewardFunction
from ..utils.utils import get_AO_TA_R

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
            pince_reward = self.pincer_attack_function(env, agent_id)

            new_reward += safe_dist_reward + help_reward + pince_reward

        return self._process(new_reward, agent_id)

    def safe_dist_function(self, version):
        """
        和队友保持安全距离的奖励函数，R单位为km
        """
        if version == 'v0':
            return lambda R: -5. * max(0., (self.min_dist - R) / self.min_dist)
        else:
            raise NotImplementedError(f"未知的队友安全距离函数版本: {version}")

    def help_function(self, version):
        """
        队友支援的奖励函数，R单位为km
        """
        if version == 'v0':
            return lambda R: 0.4 * np.exp(-(R - self.opt_attack_dist) ** 2 / self.dist_var)
        else:
            raise NotImplementedError(f"未知的队友支援奖励函数版本: {version}")

    def pincer_attack_function(self, env, agent_id):
        """
        形成合围攻势的奖励函数
        """
        # 获得飞机对抗数
        num_flight = len(env.agents[agent_id].enemies)

        # 获取我机的特征
        ego_feature = np.hstack([env.agents[agent_id].get_position(),
                                 env.agents[agent_id].get_velocity()])

        # 初始化返回奖励值
        r = 0

        # 先计算我机相对敌机的相对位置
        s = 0
        for enm in env.agents[agent_id].enemies:
            enm_feature = np.hstack([enm.get_position(),
                                     enm.get_velocity()])
            _, _, _, side_flag = get_AO_TA_R(ego_feature, enm_feature, return_side=True)
            s += side_flag
        r += 1. - np.abs(s / num_flight)

        # 计算友机相对敌机的相对位置
        for partner in env.agents[agent_id].partners:
            partner_feature = np.hstack([partner.get_position(),
                                         partner.get_velocity()])
            s = 0
            for enm in partner.enemies:
                enm_feature = np.hstack([enm.get_position(),
                                         enm.get_velocity()])
                _, _, _, side_flag = get_AO_TA_R(partner_feature, enm_feature, return_side=True)
                s += side_flag
            r += 1. - np.abs(s / num_flight)

        return r

