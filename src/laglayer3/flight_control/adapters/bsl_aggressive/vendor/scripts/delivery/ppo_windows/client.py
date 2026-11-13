"""High-level real-PPO adapter for one actual platform episode.

Only target quantities are converted from the documented convenient units.
Raw platform property names and units are passed through without conversion.
No transport, environment, implicit reset, PID fallback or training is created.
"""
from copy import deepcopy
import math
from pathlib import Path

import numpy as np

from ....scripts.delivery.ppo_windows.inference import actor_from_package


def _finite(value,name):
    if isinstance(value,bool) or not isinstance(value,(int,float,np.number)) or not math.isfinite(value):
        raise ValueError(name+" must be a finite real number")
    return float(value)


def _reference(*,target_heading_deg,target_height_m,target_ground_speed_mps,
        gamma_ref_deg=0.,heading_rate_degps=0.,climb_rate_mps=0.,speed_rate_mps2=0.):
    heading=_finite(target_heading_deg,"target_heading_deg")
    height=_finite(target_height_m,"target_height_m")
    speed=_finite(target_ground_speed_mps,"target_ground_speed_mps")
    if speed<=0.:raise ValueError("target_ground_speed_mps must be positive; it is not EAS")
    return dict(heading_rad=math.radians(heading%360.),height_m=height,ground_speed_mps=speed,
        gamma_ref_rad=math.radians(_finite(gamma_ref_deg,"gamma_ref_deg")),
        rates=[_finite(heading_rate_degps,"heading_rate_degps"),_finite(climb_rate_mps,"climb_rate_mps"),
            _finite(speed_rate_mps2,"speed_rate_mps2")],preview_gamma_rates_degps=None)


class PPOFlightController:
    """Load a bound Actor and execute one explicitly reset real episode.

    reset initializes the genuine first raw/reference but does not infer.
    step at logical0 consumes that same observation; later step calls use the
    actual next observations at0.2-second spacing. Create a new client for a
    new actual episode; duplicate reset on this client is rejected.
    """
    def __init__(self,package_root):
        self.actor,self.manifest=actor_from_package(Path(package_root).resolve())
        self._initialized=False;self._faulted=False;self._last_action_time=None
        self._initial_raw=None;self._initial_reference=None

    def reset(self,raw,*,target_heading_deg,target_height_m,target_ground_speed_mps,
            logical_time_sec=0.,gamma_ref_deg=0.,heading_rate_degps=0.,climb_rate_mps=0.,speed_rate_mps2=0.):
        if self._initialized or self._faulted:
            raise RuntimeError("one explicit reset per client/real episode; create a new client for a new episode")
        time=_finite(logical_time_sec,"logical_time_sec")
        if time!=0.:raise ValueError("the first real episode observation requires logical_time_sec=0")
        ref=_reference(target_heading_deg=target_heading_deg,target_height_m=target_height_m,
            target_ground_speed_mps=target_ground_speed_mps,gamma_ref_deg=gamma_ref_deg,
            heading_rate_degps=heading_rate_degps,climb_rate_mps=climb_rate_mps,speed_rate_mps2=speed_rate_mps2)
        try:self.actor.reset(raw,ref,time)
        except Exception:
            self._faulted=True;raise
        self._initial_raw=deepcopy(raw);self._initial_reference=deepcopy(ref);self._initialized=True

    def step(self,raw,*,logical_time_sec,target_heading_deg,target_height_m,target_ground_speed_mps,
            gamma_ref_deg=0.,heading_rate_degps=0.,climb_rate_mps=0.,speed_rate_mps2=0.,return_diagnostics=False):
        if not self._initialized:raise RuntimeError("explicit reset with the real first frame is required")
        if self._faulted:raise RuntimeError("failed control stream cannot continue or silently reset")
        time=_finite(logical_time_sec,"logical_time_sec")
        ref=_reference(target_heading_deg=target_heading_deg,target_height_m=target_height_m,
            target_ground_speed_mps=target_ground_speed_mps,gamma_ref_deg=gamma_ref_deg,
            heading_rate_degps=heading_rate_degps,climb_rate_mps=climb_rate_mps,speed_rate_mps2=speed_rate_mps2)
        if self._last_action_time is None:
            if time!=0. or raw!=self._initial_raw or ref!=self._initial_reference:
                raise ValueError("first step must consume the same logical0 raw/target supplied to reset")
        elif abs(time-(self._last_action_time+.2))>1e-9:
            raise ValueError("one real observation and action per0.2-second logical step required")
        try:
            if self._last_action_time is not None:self.actor.observe(raw,ref,time)
            result=self.actor.act()
            command=np.asarray(result["command"],dtype=np.float32)
            if command.shape!=(4,) or not np.isfinite(command).all():
                raise ValueError("actual composed/QP physical command must have four finite channels")
            self._last_action_time=time
        except Exception:
            self._faulted=True;raise
        command=command.copy()
        if return_diagnostics:
            diagnostics=deepcopy(result)
            diagnostics["high_level_api"]=dict(command_order=["aileron","elevator","rudder","throttle"],
                command_kind="physical_normalized_QP_output_not_PPO_residual",raw_units_unchanged=True,
                target_speed_kind="ground_speed_magnitude_mps_not_EAS",implicit_reset=False,PID_fallback=False)
            return command,diagnostics
        return command
