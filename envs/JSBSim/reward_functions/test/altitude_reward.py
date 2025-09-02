import numpy as np
from envs.JSBSim.reward_functions.reward_function_base import BaseRewardFunction  # 从基础奖励类继承


class AltitudeReward(BaseRewardFunction):
    """
    高度相关奖励函数
    当战斗机低于安全高度时施加惩罚（负奖励）
    包含两个惩罚项：
    - 速度惩罚 (Pv)：当高度低于安全高度时，根据下降速度计算惩罚
    - 高度惩罚 (PH)：当高度低于危险高度时，根据高度值计算惩罚
    """
    def __init__(self, config):
        super().__init__(config)
        # 安全高度阈值（单位：公里）
        self.safe_altitude = getattr(self.config, f'{self.__class__.__name__}_safe_altitude', 4.0)
        # 危险高度阈值（单位：公里）
        self.danger_altitude = getattr(self.config, f'{self.__class__.__name__}_danger_altitude', 3.5)
        # 速度惩罚系数
        self.Kv = getattr(self.config, f'{self.__class__.__name__}_Kv', 0.2)

        # 奖励项名称（用于记录）
        self.reward_item_names = [self.__class__.__name__ + item for item in ['', '_Pv', '_PH']]

    def get_reward(self, task, env, agent_id):
        """
        计算当前智能体的高度相关奖励
        
        参数:
            task: 当前任务实例
            env: 环境实例
            agent_id: 智能体ID
            
        返回:
            float: 计算出的奖励值
        """
        # 获取当前高度（转换为公里）
        ego_z = env.agents[agent_id].get_position()[-1] / 1000
        # 获取当前垂直速度（转换为马赫数）
        ego_vz = env.agents[agent_id].get_velocity()[-1] / 340
        
        # 1. 速度惩罚计算
        Pv = 0.
        if ego_z <= self.safe_altitude:
            # 基于高度差和下降速度计算惩罚
            Pv = -np.clip(ego_vz / self.Kv * (self.safe_altitude - ego_z) / self.safe_altitude, 0., 1.)
        
        # 2. 高度惩罚计算
        PH = 0.
        if ego_z <= self.danger_altitude:
            # 高度越低惩罚越大
            PH = np.clip(ego_z / self.danger_altitude, 0., 1.) - 1. - 1.
        
        # 总奖励 = 速度惩罚 + 高度惩罚
        new_reward = Pv + PH

        # file_path = "/mnt/d/FastProjects/ModelFlight/LAG/scripts/results/log/reward/altitude_reward.txt"
        # with open(file_path, "a", encoding="utf-8") as f:
        #     f.write(str(Pv) + ", " + str(PH) + "\n")

        return self._process(new_reward, agent_id)