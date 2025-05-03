import numpy as np
from wandb import agent
from ..reward_function_base import BaseRewardFunction
from ...utils.utils import get_AO_TA_R

class TeamPostureReward(BaseRewardFunction):
    def __init__(self, config):
        super().__init__(config)
        self.min_dist = getattr(self.config, f'{self.__class__.__name__}_min_dist', 40)
        self.max_dist = getattr(self.config, f'{self.__class__.__name__}_max_dist', 15)

    def get_reward(self, task, env, agent_id):
        new_reward = 0
        ego_feature = np.hstack([env.agents[agent_id].get_position(),
                                 env.agents[agent_id].get_velocity()])

        for par in env.agents[agent_id].partners:
            print("par: ", par)

        return 0