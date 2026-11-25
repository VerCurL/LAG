"""LAGLayer3 seam for the copied bsl_aggressive controller."""

from pathlib import Path
from typing import Mapping, Optional

import numpy as np

from laglayer3.domain.models import ExecutionAction


class BslAggressiveAdapter:
    """Translate LAGLayer3 execution actions into the copied BSL controller."""

    def __init__(self, model_path: Optional[str] = None) -> None:
        from .controller import AggressiveBSLFlightController

        root = Path(__file__).resolve().parent / "bsl_aggressive"
        path = Path(model_path) if model_path else root / "vendor" / "assets" / "actor_best.pt"
        settings = {
            "delta_altitude_m": (100.0, 0.0, -100.0),
            "delta_heading_deg": (-30.0, -15.0, 0.0, 15.0, 30.0),
            "delta_speed_mps": (17.0, 0.0, -17.0),
        }
        self.controller = AggressiveBSLFlightController(
            feet_to_meters=0.3048,
            model_path=str(path),
            settings=settings,
        )

    def reset(self) -> None:
        self.controller.reset()

    def control(self, aircraft_id: str, platform_state: Mapping[str, object], action: ExecutionAction):
        action4 = np.asarray([action.altitude, action.heading, action.speed, 0], dtype=np.int64)
        output = self.controller.controls([aircraft_id], [dict(platform_state)], [action4])
        return np.asarray(output[0], dtype=np.float32)

    def forget(self, aircraft_id: str) -> None:
        self.controller.forget(aircraft_id)
