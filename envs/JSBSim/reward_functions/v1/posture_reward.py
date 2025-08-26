import numpy as np
import math
from wandb import agent
from envs.JSBSim.reward_functions.reward_function_base import BaseRewardFunction
from envs.JSBSim.utils.utils import get_AO_TA_R  # 导入角度计算工具


class PostureReward(BaseRewardFunction):
    """
    战斗机姿态相关奖励函数
    奖励 = 方向奖励 * 距离奖励
    - 方向奖励：鼓励指向敌机，惩罚被敌机指向
    - 距离奖励：鼓励接近敌机，惩罚过远
    
    注意：当前仅支持1v1对抗环境
    """
    def __init__(self, config):
        super().__init__(config)
        # 方向奖励计算版本
        self.orientation_version = getattr(self.config, f'{self.__class__.__name__}_orientation_version', 'v2')
        # 距离奖励计算版本
        self.range_version = getattr(self.config, f'{self.__class__.__name__}_range_version', 'v3')

        # 理想目标距离（单位：公里）
        self.target_dist = getattr(self.config, f'{self.__class__.__name__}_target_dist', 3.0)
        # 理想攻击角度（单位：弧度 ）
        self.attack_angle = getattr(self.config, f'{self.__class__.__name__}_attack_angle', 45)
        self.attack_angle = math.radians(self.attack_angle)

        # 获取方向奖励计算函数
        self.orientation_fn = self.get_orientation_function(self.orientation_version)
        # 获取距离奖励计算函数
        self.range_fn = self.get_range_function(self.range_version)

        # 奖励项名称（用于记录）
        self.reward_item_names = [self.__class__.__name__ + item for item in ['', '_orn', '_range']]

    def get_reward(self, task, env, agent_id):
        """
        计算当前智能体的姿态相关奖励
        
        参数:
            task: 当前任务实例
            env: 环境实例
            agent_id: 智能体ID
            
        返回:
            float: 计算出的奖励值
        """
        new_reward = 0
        # 获取自身状态：位置(北,东,下) + 速度(北,东,下)
        ego_feature = np.hstack([env.agents[agent_id].get_position(),
                                 env.agents[agent_id].get_velocity()])
        
        # 遍历所有敌人
        for enm in env.agents[agent_id].enemies:
            # 获取敌人状态
            enm_feature = np.hstack([enm.get_position(),
                                    enm.get_velocity()])
            # 计算角度关系：AO(攻击角，弧度), TA(威胁角，弧度), R(距离，米)
            AO, TA, R = get_AO_TA_R(ego_feature, enm_feature)

            # 计算方向奖励（基于AO和TA）
            orientation_reward = self.orientation_fn(AO, TA)
            # 计算距离奖励（转换为公里）
            range_reward = self.range_fn(R / 1000)

            # 总奖励 = 方向奖励 * 距离奖励
            new_reward += orientation_reward * range_reward

        return self._process(new_reward, agent_id, (orientation_reward, range_reward))

    def get_orientation_function(self, version):
        """根据版本选择方向奖励计算函数"""
        if version == 'v0':
            # 版本0：使用双曲正切函数计算AO和TA
            return lambda AO, TA: (1. - np.tanh(9 * (AO - np.pi / 9))) / 3. + 1 / 3. \
                + min((np.arctanh(1. - max(2 * TA / np.pi, 1e-4))) / (2 * np.pi), 0.) + 0.5
        elif version == 'v1':
            # 版本1：AO和TA的乘积形式
            return lambda AO, TA: (1. - np.tanh(2 * (AO - np.pi / 2))) / 2. \
                * (np.arctanh(1. - max(2 * TA / np.pi, 1e-4))) / (2 * np.pi) + 0.5
        elif version == 'v2':
            # 版本2：AO的线性惩罚 + TA的对数惩罚
            return lambda AO, TA: 1 / (50 * AO / np.pi + 2) + 1 / 2 \
                + min((np.arctanh(1. - max(2 * TA / np.pi, 1e-4))) / (2 * np.pi), 0.) + 0.5
        else:
            raise NotImplementedError(f"未知的方向函数版本: {version}")

    def get_range_function(self, version):
        """根据版本选择距离奖励计算函数"""
        if version == 'v0':
            # 版本0：高斯分布+逻辑函数组合
            return lambda R: np.exp(-(R - self.target_dist) ** 2 * 0.004) / (1. + np.exp(-(R - self.target_dist + 2) * 2))
        elif version == 'v1':
            # 版本1：指数衰减+逻辑函数组合
            return lambda R: np.clip(1.2 * np.min([np.exp(-(R - self.target_dist) * 0.21), 1]) /
                                     (1. + np.exp(-(R - self.target_dist + 1) * 0.8)), 0.3, 1)
        elif version == 'v2':
            # 版本2：在v1基础上增加符号函数
            return lambda R: max(np.clip(1.2 * np.min([np.exp(-(R - self.target_dist) * 0.21), 1]) /
                                         (1. + np.exp(-(R - self.target_dist + 1) * 0.8)), 0.3, 1), np.sign(7 - R))
        elif version == 'v3':
            # 版本3：分段函数（近距离恒定+中距离二次函数+远距离指数衰减）
            return lambda R: 1 * (R < 5) + (R >= 5) * np.clip(-0.032 * R**2 + 0.284 * R + 0.38, 0, 1) + np.clip(np.exp(-0.16 * R), 0, 0.2)
        else:
            raise NotImplementedError(f"未知的距离函数版本: {version}")