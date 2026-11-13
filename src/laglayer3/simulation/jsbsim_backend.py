"""Small JSBSim aircraft backend used behind LocalJSBSimAdapter."""

import math
from pathlib import Path
from typing import Mapping, Tuple

import jsbsim

from laglayer3.domain.models import AircraftState, ExecutionAction


FT_TO_M = 0.3048


class JSBSimAircraft:
    def __init__(self, aircraft: AircraftState, data_root: str, model: str = "f16", dt: float = 1.0 / 60.0):
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
        # This direct mapping is a contract smoke path. The production path
        # calls BslAggressiveAdapter before these platform properties.
        values = {
            "fcs/aileron-cmd-norm": (action.heading - 2) / 2.0,
            "fcs/elevator-cmd-norm": (action.altitude - 1) / 1.0,
            "fcs/rudder-cmd-norm": 0.0,
            "fcs/throttle-cmd-norm": 0.4 + action.speed * 0.25,
        }
        for name, value in values.items():
            self.fdm.set_property_value(name, float(max(-1.0, min(1.0, value))))

    def step(self, action: ExecutionAction) -> AircraftState:
        self.command(action)
        if not self.fdm.run():
            raise RuntimeError("JSBSim aircraft step failed")
        return self.state()

    def state(self) -> AircraftState:
        get = self.fdm.get_property_value
        speed = math.sqrt(
            sum(get(name) ** 2 for name in ("velocities/v-north-fps", "velocities/v-east-fps", "velocities/v-down-fps"))
        ) * FT_TO_M
        heading = math.degrees(get("attitude/psi-true-rad")) % 360.0
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
            metadata={"backend": "jsbsim", "model": self.model},
        )
