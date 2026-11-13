"""Independent aggressive BSL fork with faster references and explicit power.

The copied nominal/residual/QP chain owns final A/E/R/T. High-level accelerate
requests a 0.9 throttle proposal before QP; every aircraft keeps its own state.
"""

from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
from threading import Lock
from time import perf_counter

import numpy as np
import torch

_LOAD_LOCK = Lock()


@lru_cache(maxsize=4)
def _load_policy(model_path):
    root = Path(__file__).resolve().parent
    manifest = json.loads((root/"source_manifest.json").read_text(encoding="utf-8"))
    for name, identity in manifest["files"].items():
        path = root/"vendor"/name
        if hashlib.sha256(path.read_bytes()).hexdigest() != identity["integrated_sha256"]:
            raise ValueError(f"Integrated BSL runtime changed: {name}")
    from .vendor.competition.flight_control_ppo_delivery.actor import load_policy
    from .vendor.competition.flight_control_ppo_delivery.config import load_config
    expected = manifest["files"]["assets/actor_best.pt"]["original_sha256"]
    policy = load_policy(Path(model_path), expected)
    policy.actor.requires_grad_(False)
    policy.critic.requires_grad_(False)
    return policy, load_config()


class AggressiveBSLFlightController:
    def __init__(self, feet_to_meters, model_path=None, actor=None, settings=None, diagnostics=True):
        if settings is None:
            from algorithms.utils.action_spec import ACTION_NAMES, FLIGHT_CONTROLLER
            self.action_names, self.settings = ACTION_NAMES, FLIGHT_CONTROLLER
        else:
            self.settings = settings
            self.action_names = ("altitude", "turn", "speed", "weapon")
        self.ft = float(feet_to_meters)
        self.diagnostics = bool(diagnostics)
        if not math.isclose(self.ft, .3048, rel_tol=0, abs_tol=1e-12):
            raise ValueError("BSL native field conversion requires 0.3048 meters per foot")
        self.model_path = Path(model_path) if model_path is not None else Path(__file__).parent/"vendor/assets/actor_best.pt"
        # Only immutable weights/config are shared. Controllers, filters, QP
        # history and GRU state belong to each aircraft in each environment.
        with _LOAD_LOCK:
            self.policy, self.configuration = _load_policy(str(self.model_path.resolve()))
        self.timings = dict.fromkeys(("flight_observation_seconds", "flight_actor_seconds",
                                     "flight_control_seconds", "flight_diagnostics_seconds",
                                     "flight_actor_batches", "flight_aircraft_steps"), 0.0)
        self.reset()

    def reset(self):
        self.aircraft = {}
        self.control_steps = {}
        self.last_diagnostics = {}
        self.last_references = {}

    def forget(self, name):
        self.aircraft.pop(name, None)
        self.control_steps.pop(name, None)
        self.last_diagnostics.pop(name, None)
        self.last_references.pop(name, None)

    def deltas(self, action):
        return tuple(float(self.settings[key][int(action[self.action_names.index(head)])])
                     for head, key in (("altitude", "delta_altitude_m"),
                                       ("turn", "delta_heading_deg"), ("speed", "delta_speed_mps")))

    def reference(self, plane, action):
        from .vendor.competition.flight_control_ppo_delivery.controller import _finite
        dh, dpsi, dv = self.deltas(action)
        if plane.get("control/assault-enabled", False) and not plane.get("control/policy-owned", False) and dpsi:
            dpsi = math.copysign(30.0, dpsi)
        height = _finite(plane["position/h-sl-ft"], "height")*self.ft
        heading = _finite(plane["attitude/psi-deg"], "heading")
        ground = math.sqrt(sum((_finite(plane[f"velocities/v-{axis}-fps"], axis)*self.ft)**2
                               for axis in ("north", "east", "down")))
        speed = ground+dv
        speed_limit=plane.get('control/speed-limit-mps')
        if speed_limit is not None:
            speed=min(speed,float(speed_limit)-20.)
        cruise_cas=plane.get('control/navigation-cas-mps')
        if cruise_cas is not None:
            from .airspeed import calibrated_airspeed_mps
            speed=max(80.,ground*float(cruise_cas)/max(1.,calibrated_airspeed_mps(plane,self.ft)))
        if speed <= 0:
            raise ValueError("BSL requested ground speed must remain positive")
        return {"heading_rad": math.radians(float(plane.get('control/navigation-heading-deg',heading+dpsi)) % 360),
                "height_m": float(plane.get("control/altitude-target-m", height+dh)),
                "ground_speed_mps": speed, "gamma_ref_rad": 0.0,
                # These are public target steps, not inferred continuous rates.
                "rates": [0.0, 0.0, 0.0], "preview_gamma_rates_degps": None}

    def controls(self, names, planes, actions):
        return self.controls_many([(self, names, planes, actions)])[0]

    def _prepare(self, name, plane, action):
        from .vendor.competition.flight_control_ppo_delivery.actor import ActorDelivery
        from .vendor.scripts.delivery.ppo_windows.guarded_controller import GuardedController
        started = perf_counter()
        ref = self.reference(plane, action)
        logical_time = self.control_steps.get(name, 0)*.2
        if name not in self.aircraft:
            if self.diagnostics:
                controller = GuardedController(self.configuration)
            else:
                from .fast_runtime import LeanGuardedController
                controller = LeanGuardedController(self.configuration)
            delivery = ActorDelivery(self.policy, controller)
            delivery.reset(plane, ref, 0.0)
            self.aircraft[name] = delivery
        else:
            delivery = self.aircraft[name]
            delivery.observe(plane, ref, logical_time)
        self.timings["flight_observation_seconds"] += perf_counter()-started
        self.last_references[name] = ref
        delivery.controller._assault_power = bool(plane.get("control/assault-enabled", False)) and plane.get('control/navigation-cas-mps') is None
        speed_limit=plane.get('control/speed-limit-mps')
        if speed_limit is None:delivery.controller._speed_throttle_ceiling=1.
        else:
            speed_true=math.sqrt(sum(float(plane.get(f'velocities/{x}-fps',0))**2 for x in ('u','v','w')))*self.ft
            delivery.controller._speed_throttle_ceiling=float(np.clip(.55+.035*(float(speed_limit)-20.-speed_true),0.,1.))
        delivery.controller._aggressive_accelerate = int(action[self.action_names.index("speed")]) == 0
        return delivery

    def _apply(self, name, delivery, residual, post):
        started = perf_counter()
        # Copies of hidden states are useful for a full replay but not needed
        # for the training control path. Save them before committing state.
        before = delivery.hidden.copy() if self.diagnostics else None
        mask_before = delivery.mask.copy() if self.diagnostics else None
        self.timings["flight_diagnostics_seconds"] += perf_counter()-started
        started = perf_counter()
        result = delivery.controller.apply(residual)
        command = np.asarray(result["command"], dtype=np.float64)
        if command.shape != (4,) or not np.isfinite(command).all():
            raise ValueError("BSL must return four finite final platform commands")
        delivery.actor_forward_evaluations += 1
        delivery.hidden = post.copy()
        delivery.mask.fill(1.)
        self.control_steps[name] = self.control_steps.get(name, 0)+1
        self.timings["flight_control_seconds"] += perf_counter()-started
        self.timings["flight_aircraft_steps"] += 1
        started = perf_counter()
        if self.diagnostics:
            result.update(actor_state_before=before, actor_state_post=delivery.hidden.copy(), mask_before=mask_before,
                          deterministic=True, actor_forward_evaluations=delivery.actor_forward_evaluations,
                          critic_forward_evaluations=0, PID_fallback=False)
            self.last_diagnostics[name] = result
        self.timings["flight_diagnostics_seconds"] += perf_counter()-started
        return command

    @staticmethod
    def controls_many(requests):
        return list(AggressiveBSLFlightController.iter_controls_many(requests))

    @staticmethod
    def iter_controls_many(requests):
        """Infer once per shared policy; apply/QP and yield each request in order.

        Callers may submit a completed environment's I/O while this generator
        continues applying the other aircraft's already computed residuals.
        No next-state observation is consumed until the caller's step barrier.
        """
        requests = list(requests)
        seen = set()
        groups, prepared = {}, []
        # Validate all identities before advancing any per-aircraft history.
        for controller, names, planes, actions in requests:
            if len(names) != len(planes) or len(names) != len(actions):
                raise ValueError("BSL requests need matching names, planes and actions")
            for name in names:
                key = (id(controller), name)
                if key in seen:
                    raise ValueError("Aircraft occurs twice in a BSL control batch")
                seen.add(key)
        for controller, names, planes, actions in requests:
            rows = []
            for name, plane, action in zip(names, planes, actions):
                delivery = controller._prepare(name, plane, action)
                row = [controller, name, delivery, None, None]
                groups.setdefault(id(delivery.policy), []).append(row)
                rows.append(row)
            prepared.append(rows)
        for rows in groups.values():
            started = perf_counter()
            observations = np.stack([r[2].controller._observation for r in rows])
            hidden = np.concatenate([r[2].hidden for r in rows], axis=0)
            masks = np.concatenate([r[2].mask for r in rows], axis=0)
            with torch.no_grad():
                residuals, post = rows[0][2].policy.act(observations, hidden, masks, deterministic=True)
            residuals = residuals.detach().cpu().numpy()
            post = post.detach().cpu().numpy()
            # Attribute shared work once so aggregate timers do not double count.
            rows[0][0].timings["flight_actor_seconds"] += perf_counter()-started
            rows[0][0].timings["flight_actor_batches"] += 1
            for row, residual, state in zip(rows, residuals, post):
                row[3], row[4] = residual, state[None, ...]
        for rows in prepared:
            output = np.empty((len(rows), 4), dtype=np.float64)
            for index, (controller, name, delivery, residual, post) in enumerate(rows):
                output[index] = controller._apply(name, delivery, residual, post)
            yield output
