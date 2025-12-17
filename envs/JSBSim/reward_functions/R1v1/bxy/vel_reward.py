import numpy as np
from envs.JSBSim.reward_functions.reward_function_base import BaseRewardFunction
from envs.JSBSim.core.catalog import Catalog as c

change = 0.3048
class VelReward(BaseRewardFunction):
    """
    SelfSpeedReward
    Punish or reward based on the value of selfV.
    """
    def __init__(self, config):
        super().__init__(config)
        self.reward_item_names = [self.__class__.__name__]

    def get_reward(self, task, env, agent_id):
        """
        Reward calculation based on selfV.

        Args:
            task: task instance
            env: environment instance

        Returns:
            (float): reward
        """

        # 转为马赫数！
        ego_v = env.agents[agent_id].get_property_value(c.velocities_vt_fps) * change / 304.
        enm_v = env.agents[agent_id].enemies[0].get_property_value(c.velocities_vt_fps) * change / 304.

        def calculate_Rv(v_opt, v_R, v_B):

            if v_opt < v_R:
                Rv = np.exp((v_opt - v_R) / v_opt)
            elif 0.5 * v_B < v_R <= v_opt:
                Rv = 0.1 + 0.9 * (v_R - 0.5 * v_B) / (v_opt - 0.5 * v_B)
            else:
                Rv = 0.1
            return Rv

        # 示例：最佳空战速度 v_opt * 340 （转换成马赫）220m/s
        v_opt = 0.65
        reward = calculate_Rv(v_opt, ego_v, enm_v)

        return self._process(reward, agent_id)
