import math
from .reward_function_base import BaseRewardFunction
from ..core.catalog import Catalog as c


class HeadingReward(BaseRewardFunction):
    """
    Measure the difference between the current heading and the target heading
    """
    def __init__(self, config):
        super().__init__(config)
        cls = self.__class__.__name__
        self.kernel = str(getattr(config, f"{cls}_kernel", "legacy"))
        self.heading_sigma_deg = float(getattr(config, f"{cls}_heading_sigma_deg", 20.0))
        self.heading_gauss_weight = float(getattr(config, f"{cls}_heading_gauss_weight", 0.55))
        self.heading_cos_weight = float(getattr(config, f"{cls}_heading_cos_weight", 0.45))
        self.altitude_sigma_m = float(getattr(config, f"{cls}_altitude_sigma_m", 150.0))
        self.speed_sigma_mps = float(getattr(config, f"{cls}_speed_sigma_mps", 35.0))
        self.roll_sigma_rad = float(getattr(config, f"{cls}_roll_sigma_rad", 0.45))
        self.track_heading_weight = float(getattr(config, f"{cls}_track_heading_weight", 0.40))
        self.track_altitude_weight = float(getattr(config, f"{cls}_track_altitude_weight", 0.25))
        self.track_speed_weight = float(getattr(config, f"{cls}_track_speed_weight", 0.25))
        self.track_roll_weight = float(getattr(config, f"{cls}_track_roll_weight", 0.10))
        self.normalize_track_weights = bool(getattr(config, f"{cls}_normalize_track_weights", True))
        self.roll_relief_enable = bool(getattr(config, f"{cls}_roll_relief_enable", False))
        self.roll_relief_heading_deg = float(getattr(config, f"{cls}_roll_relief_heading_deg", 35.0))
        self.roll_relief_strength = float(getattr(config, f"{cls}_roll_relief_strength", 0.75))
        self.roll_min_weight_factor = float(getattr(config, f"{cls}_roll_min_weight_factor", 0.25))
        self.reward_item_names = [
            self.__class__.__name__,
            self.__class__.__name__ + "_heading",
            self.__class__.__name__ + "_alt",
            self.__class__.__name__ + "_roll",
            self.__class__.__name__ + "_speed",
            self.__class__.__name__ + "_heading_gauss",
            self.__class__.__name__ + "_heading_cos",
            self.__class__.__name__ + "_rho_alt",
            self.__class__.__name__ + "_rho_speed",
            self.__class__.__name__ + "_rho_roll",
            self.__class__.__name__ + "_norm_heading",
            self.__class__.__name__ + "_norm_alt",
            self.__class__.__name__ + "_norm_speed",
            self.__class__.__name__ + "_norm_roll",
        ]

    @staticmethod
    def _pseudo_huber(x):
        return math.sqrt(1.0 + x * x) - 1.0

    def _get_track_weights(self, roll_weight=None):
        if roll_weight is None:
            roll_weight = self.track_roll_weight
        weights = [
            self.track_heading_weight,
            self.track_altitude_weight,
            self.track_speed_weight,
            float(roll_weight),
        ]
        if not self.normalize_track_weights:
            return weights
        total = sum(weights)
        if total <= 1e-8:
            return [0.40, 0.25, 0.25, 0.10]
        return [w / total for w in weights]

    def _get_roll_weight(self, delta_heading):
        if not self.roll_relief_enable:
            return self.track_roll_weight
        heading_scale = max(self.roll_relief_heading_deg, 1e-6)
        relief = min(abs(float(delta_heading)) / heading_scale, 1.0)
        min_factor = min(max(self.roll_min_weight_factor, 0.0), 1.0)
        strength = min(max(self.roll_relief_strength, 0.0), 1.0)
        factor = max(min_factor, 1.0 - strength * relief)
        return self.track_roll_weight * factor

    def _get_heading_weights(self):
        wg = max(self.heading_gauss_weight, 0.0)
        wc = max(self.heading_cos_weight, 0.0)
        total = wg + wc
        if total <= 1e-8:
            return 0.55, 0.45
        return wg / total, wc / total

    @staticmethod
    def _get_speed_error_mps(task, env, agent_id):
        if hasattr(task, "get_speed_error_mps"):
            return float(task.get_speed_error_mps(env.agents[agent_id]))
        return float(
            env.agents[agent_id].get_property_value(c.delta_velocities_u)
        )

    def get_reward(self, task, env, agent_id):
        """
        Reward is built as a geometric mean of scaled gaussian rewards for each relevant variable

        Args:
            task: task instance
            env: environment instance

        Returns:
            (float): reward
        """

        delta_heading = float(env.agents[agent_id].get_property_value(c.delta_heading))
        delta_altitude = float(env.agents[agent_id].get_property_value(c.delta_altitude))
        delta_speed = self._get_speed_error_mps(task, env, agent_id)
        roll_rad = float(env.agents[agent_id].get_property_value(c.attitude_roll_rad))

        if self.kernel != "smooth":
            heading_error_scale = 5.0  # degrees
            heading_r = math.exp(-((delta_heading / heading_error_scale) ** 2))
            alt_error_scale = 15.24  # m
            alt_r = math.exp(-((delta_altitude / alt_error_scale) ** 2))
            roll_error_scale = 0.35  # radians ~= 20 degrees
            roll_r = math.exp(-((roll_rad / roll_error_scale) ** 2))
            speed_error_scale = 24  # mps (~10%)
            speed_r = math.exp(-((delta_speed / speed_error_scale) ** 2))
            reward = (heading_r * alt_r * roll_r * speed_r) ** (1 / 4)
            return self._process(
                reward,
                agent_id,
                (
                    heading_r,
                    alt_r,
                    roll_r,
                    speed_r,
                    heading_r,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    delta_heading / heading_error_scale,
                    delta_altitude / alt_error_scale,
                    delta_speed / speed_error_scale,
                    roll_rad / roll_error_scale,
                ),
            )

        heading_sigma = max(self.heading_sigma_deg, 1e-6)
        alt_sigma = max(self.altitude_sigma_m, 1e-6)
        speed_sigma = max(self.speed_sigma_mps, 1e-6)
        roll_sigma = max(self.roll_sigma_rad, 1e-6)
        norm_heading = delta_heading / heading_sigma
        norm_alt = delta_altitude / alt_sigma
        norm_speed = delta_speed / speed_sigma
        norm_roll = roll_rad / roll_sigma

        heading_gauss = math.exp(-(norm_heading ** 2))
        heading_cos = (1.0 + math.cos(math.radians(delta_heading))) * 0.5
        wg, wc = self._get_heading_weights()
        heading_r = wg * heading_gauss + wc * heading_cos

        rho_alt = self._pseudo_huber(norm_alt)
        rho_speed = self._pseudo_huber(norm_speed)
        rho_roll = self._pseudo_huber(norm_roll)
        alt_r = math.exp(-rho_alt)
        speed_r = math.exp(-rho_speed)
        roll_r = math.exp(-rho_roll)

        w_psi, w_h, w_v, w_phi = self._get_track_weights(self._get_roll_weight(delta_heading))
        reward = w_psi * heading_r + w_h * alt_r + w_v * speed_r + w_phi * roll_r
        return self._process(
            reward,
            agent_id,
            (
                heading_r,
                alt_r,
                roll_r,
                speed_r,
                heading_gauss,
                heading_cos,
                rho_alt,
                rho_speed,
                rho_roll,
                norm_heading,
                norm_alt,
                norm_speed,
                norm_roll,
            ),
        )
