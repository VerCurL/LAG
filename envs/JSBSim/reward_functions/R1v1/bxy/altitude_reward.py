import numpy as np
from .reward_function_base import BaseRewardFunction


class AltitudeReward(BaseRewardFunction):
    """
    AltitudeReward
    Punish if current fighter doesn't satisfy some constraints. Typically negative.
    - Punishment of velocity when lower than safe altitude   (range: [-1, 0])
    - Punishment of altitude when lower than danger altitude (range: [-1, 0])
    """
    def __init__(self, config):
        super().__init__(config)
        self.safe_altitude = getattr(self.config, f'{self.__class__.__name__}_safe_altitude', 4.0)         # km
        self.danger_altitude = getattr(self.config, f'{self.__class__.__name__}_danger_altitude', 3.5)     # km
        self.Kv = getattr(self.config, f'{self.__class__.__name__}_Kv', 0.2)     # mh
        self.Kh = getattr(self.config, f'{self.__class__.__name__}_Kh', 2)       # km

        self.reward_item_names = [self.__class__.__name__ + item for item in ['', '_Pv', '_PH']]

    def get_reward(self, task, env, agent_id):
        """
        Reward is the sum of all the punishments.

        Args:
            task: task instance
            env: environment instance

        Returns:
            (float): reward
        """
        ego_z = env.agents[agent_id].get_position()[-1] / 1000    # unit: km
        ego_vz = env.agents[agent_id].get_velocity()[-1] / 340    # unit: mh
        enm_z = env.agents[agent_id].enemies[0].get_position()[-1] / 1000   # unit:km
        # 竖直方向的速度控制，防坠机
        Pv = 0.
        if ego_z <= self.safe_altitude:
            Pv = np.exp(-ego_vz / self.Kv * (self.safe_altitude - ego_z) / self.safe_altitude) - 1.
        # 高度数值控制，防坠机
        PH = 0.
        if ego_z <= self.danger_altitude:
            PH = np.exp(-(self.danger_altitude - ego_z) / self.danger_altitude) - 1.
        # 和敌机的相对高度控制，保持优质站位
        Prh = 0.
        relative_altitude = ego_z - enm_z
        if -2 * self.Kh <= relative_altitude < -self.Kh:
            Prh = -1. / self.Kh * relative_altitude - 2.
        elif -self.Kh <= relative_altitude < self.Kh:
            Prh = 1. / self.Kh * relative_altitude
        elif self.Kh <= relative_altitude < 2 * self.Kh:
            Prh = -1. / self.Kh * relative_altitude + 2.

        new_reward = Pv + PH + Prh
        return self._process(new_reward, agent_id, (Pv, PH))
