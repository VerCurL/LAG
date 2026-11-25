
"""固定参考航迹的单机执行层环境。"""

import math
from pathlib import Path

import numpy as np

from laglayer3.control import BslAggressiveAdapter
from laglayer3.domain.models import AircraftState, ExecutionAction, ScenarioInitialization
from laglayer3.execution.observation import encode_execution_observation
from laglayer3.planning.route_library import FixedRouteLibrary
from laglayer3.simulation import LocalJSBSimAdapter

from .tracking import RouteTrackingReward, reference_at

ACTION_DIMS = (3, 5, 3)
NEUTRAL = np.array([1, 2, 1], dtype=np.int64)


class FixedRouteExecutionEnv:
    """每次 reset 选择一条航迹，每次 step 执行一个 5 Hz 动作。"""

    def __init__(self, library, simulation, route_names, window_size, reward, seed):
        self.library = library
        self.simulation = simulation
        self.route_names = tuple(route_names)
        self.window_size = window_size
        self.reward_model = reward
        self.random = np.random.default_rng(seed)
        self.route = None
        self.done = True
        if not self.route_names or any(name not in library.routes for name in self.route_names):
            raise ValueError("训练航迹名称必须存在于缓存中")
        if isinstance(window_size, bool) or not isinstance(window_size, int) or window_size < 1:
            raise ValueError("前视窗口点数必须为正整数")
        step_count = library.horizon_seconds / simulation.control_interval_seconds
        if not math.isclose(step_count, round(step_count)):
            raise ValueError("航迹时长必须与执行周期对齐")
        self.episode_steps = round(step_count)

    def reset(self, route_name=None):
        self.route_name = self.random.choice(self.route_names) if route_name is None else route_name
        speed = self.library.initial_speed_mps
        altitude = self.library.initial_altitude_m
        initial = AircraftState("aircraft_0", "red", (0.0, 0.0, -altitude),
                                (speed, 0.0, 0.0), 0.0, altitude, speed)
        snapshot = self.simulation.reset(ScenarioInitialization("fixed_route", (initial,)))
        self.aircraft = snapshot.aircraft[0]
        self.route = self.library.bind(self.route_name, self.aircraft)
        self.reference_points = np.asarray(((0.0, 0.0, altitude, speed),) + self.route.points)
        self.previous_action = NEUTRAL.copy()
        self.elapsed = 0.0
        self.steps = 0
        self.done = False
        return self.observation()

    def observation(self):
        return encode_execution_observation(
            self.aircraft, self.route, self.elapsed, self.previous_action, self.window_size
        )

    def step(self, action):
        if self.done:
            raise RuntimeError("请在新回合开始前调用 reset")
        action = np.asarray(action)
        if action.shape != (3,) or action.dtype.kind not in "iu" or (
            (action < 0) | (action >= ACTION_DIMS)
        ).any():
            raise ValueError("执行动作须符合 MultiDiscrete([3, 5, 3])")
        snapshot = self.simulation.step(
            {self.aircraft.aircraft_id: ExecutionAction(*map(int, action))}
        )
        self.aircraft = snapshot.aircraft[0]
        self.steps += 1
        self.elapsed = self.steps * self.simulation.control_interval_seconds
        reference, heading = reference_at(self.reference_points, self.route.interval_seconds, self.elapsed)
        reward, metrics = self.reward_model.measure(
            self.aircraft, self.route, reference, heading, action, self.previous_action
        )
        self.previous_action = action.copy()
        end_of_route = self.steps >= self.episode_steps
        sim_status = self.simulation.termination_status()
        simulation_failure = sim_status.terminated and sim_status.reason != "time_limit"
        terminated = end_of_route or simulation_failure
        truncated = sim_status.terminated and not terminated
        self.done = terminated or truncated
        completed = end_of_route and not simulation_failure and self.reward_model.completed(metrics)
        if terminated:
            reward += self.reward_model.terminal_bonus if completed else -self.reward_model.terminal_bonus
        reason = sim_status.reason
        if end_of_route and not simulation_failure:
            reason = "route_completed" if completed else "route_tolerance_exceeded"
        metrics.update(route=self.route_name, elapsed_seconds=self.elapsed,
                       completed=bool(completed), reason=reason)
        return self.observation(), float(reward), terminated, truncated, metrics


def make_execution_env(config, project_root, seed):
    route = config["route"]
    simulation = config["simulation"]
    root = Path(project_root).resolve()
    library = FixedRouteLibrary(root / route["spec"], root / route["cache"])
    if simulation["backend"] != "jsbsim" or (
        simulation["simulation_hz"], simulation["execution_hz"]
    ) != (60, 5):
        raise ValueError("固定航迹训练当前要求 JSBSim 60 Hz 和 bsl_aggressive 5 Hz")
    checkpoint = simulation.get("controller_checkpoint")
    adapter = BslAggressiveAdapter(root / checkpoint if checkpoint else None)
    simulator = LocalJSBSimAdapter(
        step_seconds=1.0 / simulation["simulation_hz"],
        control_interval_seconds=1.0 / simulation["execution_hz"],
        time_limit_seconds=library.horizon_seconds,
        backend=simulation["backend"],
        data_root=str(root / simulation["data_root"]),
        model=simulation["aircraft_model"],
        control_adapter=adapter,
    )
    reward = RouteTrackingReward(**config["reward"])
    return FixedRouteExecutionEnv(library, simulator, route["names"], route["window_points"], reward, seed)
