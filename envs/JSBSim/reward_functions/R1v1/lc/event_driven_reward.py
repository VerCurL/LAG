import numpy as np

from .reward_function_base import BaseRewardFunction
from envs.JSBSim.utils.utils import LLA2NEU


class EventDrivenReward(BaseRewardFunction):
    """
    EventDrivenReward
    Achieve reward when the following event happens:
    - Shot down by missile: -200
    - Crash accidentally: -200
    - Shoot down other aircraft: +200
    """
    def __init__(self, config):
        super().__init__(config)
        self.missile_counts = 10


    def get_reward(self, task, env, agent_id):
        """
        Reward is the sum of all the events.

        Args:
            task: task instance
            env: environment instance

        Returns:
            (float): reward
        """

        reward = 0
        if task.shoot_flag:
            reward -= 20
        if np.sum(task.lock_duration[agent_id]) >= task.lock_duration[agent_id].maxlen and task.min_missile_attack_distance <= task.R_dis <= task.max_missile_attack_distance:
            reward += 1
        if env.agents[agent_id].is_shotdown:
            reward -= 200
        elif env.agents[agent_id].is_crash:
            reward -= 500

        for missile in env.agents[agent_id].launch_missiles:
            if missile.is_success:
                env.agents[agent_id].launch_missiles.remove(missile)
                reward += 200

        # if agent_id == "A0100" and env.current_step != 0:
        #     env.worksheet.write(env.current_step, 14, reward)

        return self._process(reward,agent_id)
