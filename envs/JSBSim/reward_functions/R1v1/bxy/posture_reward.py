import numpy as np
from wandb import agent
from envs.JSBSim.reward_functions.reward_function_base import BaseRewardFunction
from envs.JSBSim.utils.utils import get_AO_TA_R


class PostureReward(BaseRewardFunction):
    """
    PostureReward = Orientation * Range
    - Orientation: Encourage pointing at enemy fighter, punish when is pointed at.
    - Range: Encourage getting closer to enemy fighter, punish if too far away.

    NOTE:
    - Only support one-to-one environments.
    """
    def __init__(self, config):
        super().__init__(config)
        self.target_dist = getattr(self.config, f'{self.__class__.__name__}_target_dist', 3.0)
        self.reward_item_names = [self.__class__.__name__ + item for item in ['', '_orn', '_range']]

    def get_reward(self, task, env, agent_id):
        """
        Reward is a complex function of AO, TA and R in the last timestep.

        Args:
            task: task instance
            env: environment instance

        Returns:
            (float): reward
        """
        new_reward = 0
        # feature: (north, east, down, vn, ve, vd)
        ego_feature = np.hstack([env.agents[agent_id].get_position(),
                                 env.agents[agent_id].get_velocity()])
        for enm in env.agents[agent_id].enemies:
            enm_feature = np.hstack([enm.get_position(),
                                    enm.get_velocity()])
            AO, TA, R = get_AO_TA_R(ego_feature, enm_feature)
            orientation_reward = self.get_orientation_function(AO, TA)
            range_reward = self.get_range_function(R / 1000)
            new_reward += orientation_reward * range_reward
        return self._process(new_reward, agent_id, (orientation_reward, range_reward))

    def get_orientation_function(self, AO, TA):
        def penalty_factor(AO, TA):
            if AO <= np.pi / 2 and TA <= np.pi / 2:
                return 1
            else:
                return 0.5
        return penalty_factor(AO, TA) * ((1. - np.tanh(2 * (AO - np.pi / 2))) / 2.
                                                        * (np.arctanh(1. - max(2 * TA / np.pi, 1e-4))) / (2 * np.pi) + 0.5)

    def get_range_function(self, R):
        return np.exp(-1. * (R - 3)) * (R > 3) + 1. * (1 <= R <= 3) + 0.5 * (R < 1)
