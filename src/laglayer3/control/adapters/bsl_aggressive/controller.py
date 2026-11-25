"""Minimal frozen bsl_aggressive runtime for deployment inference."""

from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
from threading import Lock

import numpy as np

_LOAD_LOCK = Lock()


@lru_cache(maxsize=4)
def _load_policy(model_path):
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / "source_manifest.json").read_text(encoding="utf-8"))
    for name, identity in manifest["files"].items():
        path = root / "vendor" / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != identity["integrated_sha256"]:
            raise ValueError("Integrated BSL runtime changed: {}".format(name))
    from .vendor.competition.flight_control_ppo_delivery.actor import load_policy
    from .vendor.competition.flight_control_ppo_delivery.config import load_config

    expected = manifest["files"]["assets/actor_best.pt"]["original_sha256"]
    return load_policy(Path(model_path), expected), load_config()


class AggressiveBSLFlightController:
    """Per-aircraft GRU state with one Actor and one QP projection per action."""

    def __init__(self, feet_to_meters, model_path=None, settings=None):
        if settings is None:
            raise ValueError("explicit LAGLayer3 action settings are required")
        self.ft = float(feet_to_meters)
        if not math.isclose(self.ft, 0.3048, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("BSL native field conversion requires 0.3048 meters per foot")
        self.settings = settings
        self.action_names = ("altitude", "turn", "speed", "weapon")
        root = Path(__file__).resolve().parent
        self.model_path = Path(model_path) if model_path else root / "vendor" / "assets" / "actor_best.pt"
        with _LOAD_LOCK:
            self.policy, self.configuration = _load_policy(str(self.model_path.resolve()))
        self.reset()

    def reset(self):
        self.aircraft = {}
        self.control_steps = {}

    def forget(self, name):
        self.aircraft.pop(name, None)
        self.control_steps.pop(name, None)

    def deltas(self, action):
        return tuple(
            float(self.settings[key][int(action[self.action_names.index(head)])])
            for head, key in (("altitude", "delta_altitude_m"), ("turn", "delta_heading_deg"), ("speed", "delta_speed_mps"))
        )

    def reference(self, plane, action):
        from .vendor.competition.flight_control_ppo_delivery.controller import _finite

        delta_altitude, delta_heading, delta_speed = self.deltas(action)
        if plane.get("control/assault-enabled", False) and not plane.get("control/policy-owned", False) and delta_heading:
            delta_heading = math.copysign(30.0, delta_heading)
        height = _finite(plane["position/h-sl-ft"], "height") * self.ft
        heading = _finite(plane["attitude/psi-deg"], "heading")
        ground_speed = math.sqrt(
            sum((_finite(plane["velocities/v-{}-fps".format(axis)], axis) * self.ft) ** 2 for axis in ("north", "east", "down"))
        )
        speed = ground_speed + delta_speed
        speed_limit = plane.get("control/speed-limit-mps")
        if speed_limit is not None:
            speed = min(speed, float(speed_limit) - 20.0)
        cruise_cas = plane.get("control/navigation-cas-mps")
        if cruise_cas is not None:
            from .airspeed import calibrated_airspeed_mps

            speed = max(80.0, ground_speed * float(cruise_cas) / max(1.0, calibrated_airspeed_mps(plane, self.ft)))
        if speed <= 0.0:
            raise ValueError("BSL requested ground speed must remain positive")
        return {
            "heading_rad": math.radians(float(plane.get("control/navigation-heading-deg", heading + delta_heading)) % 360.0),
            "height_m": float(plane.get("control/altitude-target-m", height + delta_altitude)),
            "ground_speed_mps": speed,
            "gamma_ref_rad": 0.0,
            "rates": [0.0, 0.0, 0.0],
            "preview_gamma_rates_degps": None,
        }

    def controls(self, names, planes, actions):
        return self.controls_many([(self, names, planes, actions)])[0]

    def _prepare(self, name, plane, action):
        from .vendor.competition.flight_control_ppo_delivery.actor import ActorDelivery
        from .vendor.scripts.delivery.ppo_windows.guarded_controller import GuardedController

        reference = self.reference(plane, action)
        logical_time = self.control_steps.get(name, 0) * 0.2
        if name not in self.aircraft:
            delivery = ActorDelivery(self.policy, GuardedController(self.configuration))
            delivery.reset(plane, reference, 0.0)
            self.aircraft[name] = delivery
        else:
            delivery = self.aircraft[name]
            delivery.observe(plane, reference, logical_time)
        delivery.controller._assault_power = bool(plane.get("control/assault-enabled", False)) and plane.get("control/navigation-cas-mps") is None
        speed_limit = plane.get("control/speed-limit-mps")
        if speed_limit is None:
            delivery.controller._speed_throttle_ceiling = 1.0
        else:
            speed_true = math.sqrt(sum(float(plane.get("velocities/{}-fps".format(axis), 0.0)) ** 2 for axis in ("u", "v", "w"))) * self.ft
            delivery.controller._speed_throttle_ceiling = float(np.clip(0.55 + 0.035 * (float(speed_limit) - 20.0 - speed_true), 0.0, 1.0))
        delivery.controller._aggressive_accelerate = int(action[self.action_names.index("speed")]) == 0
        return delivery

    @staticmethod
    def controls_many(requests):
        return list(AggressiveBSLFlightController.iter_controls_many(requests))

    @staticmethod
    def iter_controls_many(requests):
        requests = list(requests)
        seen = set()
        prepared = []
        groups = {}
        for controller, names, planes, actions in requests:
            if len(names) != len(planes) or len(names) != len(actions):
                raise ValueError("BSL requests need matching names, planes and actions")
            rows = []
            for name, plane, action in zip(names, planes, actions):
                key = (id(controller), name)
                if key in seen:
                    raise ValueError("Aircraft occurs twice in a BSL control batch")
                seen.add(key)
                delivery = controller._prepare(name, plane, action)
                row = [controller, name, delivery, None, None]
                groups.setdefault(id(delivery.policy), []).append(row)
                rows.append(row)
            prepared.append(rows)
        for rows in groups.values():
            import torch

            observations = np.stack([row[2].controller._observation for row in rows])
            hidden = np.concatenate([row[2].hidden for row in rows], axis=0)
            masks = np.concatenate([row[2].mask for row in rows], axis=0)
            with torch.no_grad():
                residuals, states = rows[0][2].policy.act(observations, hidden, masks, deterministic=True)
            for row, residual, state in zip(rows, residuals.detach().cpu().numpy(), states.detach().cpu().numpy()):
                row[3], row[4] = residual, state[None, ...]
        for rows in prepared:
            output = np.empty((len(rows), 4), dtype=np.float64)
            for index, (controller, name, delivery, residual, state) in enumerate(rows):
                result = delivery.controller.apply(residual)
                command = np.asarray(result["command"], dtype=np.float64)
                if command.shape != (4,) or not np.isfinite(command).all():
                    raise ValueError("BSL must return four finite final platform commands")
                delivery.actor_forward_evaluations += 1
                delivery.hidden = state.copy()
                delivery.mask.fill(1.0)
                controller.control_steps[name] = controller.control_steps.get(name, 0) + 1
                output[index] = command
            yield output

