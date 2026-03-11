import os
from envs.JSBSim.reward_functions.reward_function_base import BaseRewardFunction


class EventDrivenReward(BaseRewardFunction):
    """
    EventDrivenReward (一次性事件奖励)

    当以下事件首次发生时，给予一次性的奖励或惩罚:
    - 被击落: -300
    - 意外坠毁: -500
    - 击落敌机: +300
    """

    def __init__(self, config):
        super().__init__(config)
        # 初始化用于跟踪已处理事件的集合
        # 确保每个事件只被奖励/惩罚一次
        self.shotdown_agents = set()
        self.crashed_agents = set()
        self.rewarded_missiles = set()
        self.critic_missiles = set()
        self.missile_avoided = set()

    def reset(self, task, env):
        """
        在每个回合开始时重置状态，清空已记录的事件。
        """
        self.shotdown_agents.clear()
        self.crashed_agents.clear()
        self.rewarded_missiles.clear()
        self.critic_missiles.clear()
        self.missile_avoided.clear()
        return super().reset(task, env)

    def get_reward(self, task, env, agent_id):
        """
        计算一次性的事件奖励。

        Args:
            task: task 实例
            env: environment 实例
            agent_id: 当前飞机的ID

        Returns:
            (float): 单步奖励值
        """
        reward = 0
        agent = env.agents[agent_id]

        # 1. 检查飞机是否被击落，并且这个事件是第一次发生
        if agent.is_shotdown and agent_id not in self.shotdown_agents:
            reward -= 40
            self.shotdown_agents.add(agent_id)

        # 2. 检查飞机是否坠毁，并且这个事件是第一次发生
        elif agent.is_crash and agent_id not in self.crashed_agents:
            reward -= 100
            self.crashed_agents.add(agent_id)

        # 3. 检查导弹是否成功命中
        for missile in agent.launch_missiles:
            # 我们用导弹的唯一ID (missile.id) 来做标识
            if missile.is_success and missile.uid not in self.rewarded_missiles:
                reward += 50
                self.rewarded_missiles.add(missile.uid)

        # 4. 如果导弹没有中，则给惩罚
        for missile in agent.launch_missiles:
            if missile.is_done and missile.target_aircraft.is_alive and missile.uid not in self.critic_missiles:
                reward -= 8
                self.critic_missiles.add(missile.uid)

        # 5. 是否躲避导弹成功
        for missile in agent.check_all_missile_warning():
            if not missile.is_alive and agent.is_alive and missile.uid not in self.missile_avoided:
                reward += 20
                self.missile_avoided.add(missile.uid)

        # 注意：原代码中的 _process 方法依然保留，用于后续处理
        return self._process(reward, agent_id)
