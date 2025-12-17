from envs.JSBSim.reward_functions.reward_function_base import BaseRewardFunction


class OverallSituationReward(BaseRewardFunction):
    """
    OverallSituationReward（我方全局态势信息奖励）

    针对每一时刻的全局信息状态给予奖励或惩罚:
    - 我方飞机数量占优势或劣势: num_ego - num_enm
    """

    def __init__(self, config):
        super().__init__(config)
        self.reward_item_names = [self.__class__.__name__ + item for item in ['', '_Num']]

    def get_reward(self, task, env, agent_id):
        """
        计算全局态势信息奖励。

        Args:
            task: task 实例
            env: environment 实例
            agent_id: 当前飞机的ID

        Returns:
            (float): 单步奖励值
        """
        egos_alive = [env.agents[agent_id].uid] + [ego.uid for ego in env.agents[agent_id].partners if ego.is_alive]
        enms_alive = [enm.uid for enm in env.agents[agent_id].enemies if enm.is_alive]
        reward = len(egos_alive) - len(enms_alive)

        # 注意：原代码中的 _process 方法依然保留，用于后续处理
        return self._process(reward, agent_id)
