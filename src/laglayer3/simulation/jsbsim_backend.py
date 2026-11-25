"""供 LocalJSBSimAdapter 调用的单机 JSBSim 后端。"""

import math
from pathlib import Path
from typing import Mapping, Tuple

import jsbsim

from laglayer3.domain.models import AircraftState, ExecutionAction


FT_TO_M = 0.3048


class JSBSimAircraft:
    """封装一架 JSBSim 飞机及其当前动力学状态。"""
    def __init__(self, aircraft: AircraftState, data_root: str, model: str = "f16", dt: float = 1.0 / 60.0):
        """加载飞机模型，设置积分步长并应用初始条件。"""
        self.aircraft = aircraft
        self.data_root = str(Path(data_root).resolve())
        self.model = model
        self.dt = float(dt)
        self.fdm = jsbsim.FGFDMExec(self.data_root)
        self.fdm.set_debug_level(0)
        if not self.fdm.load_model(model):
            raise RuntimeError("JSBSim failed to load aircraft model: {}".format(model))
        self.fdm.set_dt(self.dt)
        self._initialize()

    def _initialize(self) -> None:
        """写入初始位置、姿态和速度，并启动发动机稳态。"""
        heading = math.radians(self.aircraft.heading_deg)
        speed_fps = self.aircraft.speed_mps / FT_TO_M
        values = {
            "ic/long-gc-deg": 0.0,
            "ic/lat-geod-deg": 0.0,
            "ic/h-sl-ft": self.aircraft.altitude_m / FT_TO_M,
            "ic/psi-true-deg": self.aircraft.heading_deg,
            "ic/u-fps": speed_fps,
            "ic/v-fps": 0.0,
            "ic/w-fps": 0.0,
            "ic/p-rad_sec": 0.0,
            "ic/q-rad_sec": 0.0,
            "ic/r-rad_sec": 0.0,
            "ic/roc-fpm": 0.0,
            "ic/terrain-elevation-ft": 0.0,
        }
        for name, value in values.items():
            self.fdm.set_property_value(name, float(value))
        if not self.fdm.run_ic():
            raise RuntimeError("JSBSim failed to initialize aircraft")
        propulsion = self.fdm.get_propulsion()
        for index in range(propulsion.get_num_engines()):
            propulsion.get_engine(index).init_running()
        propulsion.get_steady_state()

    def command(self, action: ExecutionAction) -> None:
        """把离散执行动作转换为 JSBSim 的基础控制属性。

        这里的映射用于本地契约和 smoke 测试。接入完整飞控时，调用方应
        先通过飞控适配器生成最终平台控制量。
        """
        values = {
            "fcs/aileron-cmd-norm": (action.heading - 2) / 2.0,
            "fcs/elevator-cmd-norm": (action.altitude - 1) / 1.0,
            "fcs/rudder-cmd-norm": 0.0,
            "fcs/throttle-cmd-norm": 0.4 + action.speed * 0.25,
        }
        for name, value in values.items():
            self.fdm.set_property_value(name, float(max(-1.0, min(1.0, value))))

    def raw_state(self):
        """读取 bsl_aggressive 所需的实际平台字段。"""
        names = (
            "simulation/sim-time-sec", "simulation/dt", "position/h-sl-ft",
            "attitude/roll-rad", "attitude/pitch-rad", "attitude/psi-deg",
            "aero/alpha-deg", "aero/beta-deg", "velocities/p-rad_sec",
            "velocities/q-rad_sec", "velocities/r-rad_sec", "velocities/u-fps",
            "velocities/v-fps", "velocities/w-fps", "velocities/v-north-fps",
            "velocities/v-east-fps", "velocities/v-down-fps", "velocities/ve-fps",
            "accelerations/n-pilot-z-norm", "fcs/aileron-cmd-norm",
            "fcs/elevator-cmd-norm", "fcs/rudder-cmd-norm", "fcs/throttle-cmd-norm",
            "fcs/left-aileron-pos-norm", "fcs/right-aileron-pos-norm",
            "fcs/elevator-pos-norm", "fcs/rudder-pos-norm", "fcs/throttle-pos-norm",
        )
        return {name: float(self.fdm.get_property_value(name)) for name in names}

    def apply_low_level(self, command) -> None:
        names = (
            "fcs/aileron-cmd-norm", "fcs/elevator-cmd-norm",
            "fcs/rudder-cmd-norm", "fcs/throttle-cmd-norm",
        )
        for name, value in zip(names, command):
            self.fdm.set_property_value(name, float(value))

    def step(self, action: ExecutionAction) -> AircraftState:
        """推进一个 JSBSim 积分步并读取新的飞机状态。"""
        self.command(action)
        self.advance()
        return self.state()

    def advance(self, steps: int = 1) -> None:
        """保持当前控制量，推进指定数量的原生积分步。"""
        for _ in range(steps):
            if not self.fdm.run():
                raise RuntimeError("JSBSim aircraft step failed")

    def state(self) -> AircraftState:
        """读取 JSBSim 属性并转换为 AircraftState。"""
        get = self.fdm.get_property_value
        speed = math.sqrt(
            sum(get(name) ** 2 for name in ("velocities/v-north-fps", "velocities/v-east-fps", "velocities/v-down-fps"))
        ) * FT_TO_M
        heading = get("attitude/psi-deg") % 360.0
        altitude = get("position/h-sl-ft") * FT_TO_M
        velocity = tuple(get(name) * FT_TO_M for name in ("velocities/v-north-fps", "velocities/v-east-fps", "velocities/v-down-fps"))
        return AircraftState(
            aircraft_id=self.aircraft.aircraft_id,
            side=self.aircraft.side,
            position_ned_m=(
                get("position/lat-geod-deg") * 111132.0,
                get("position/long-gc-deg") * 111320.0,
                -altitude,
            ),
            velocity_ned_mps=velocity,
            heading_deg=heading,
            altitude_m=altitude,
            speed_mps=speed,
            alive=self.aircraft.alive,
            fuel=self.aircraft.fuel,
            metadata={
                "backend": "jsbsim",
                "model": self.model,
                "roll_rad": get("attitude/roll-rad"),
                "pitch_rad": get("attitude/pitch-rad"),
            },
        )
