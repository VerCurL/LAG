"""Shared raw -> observable40 -> original nominal/residual/QP -> sendable ZOH.

This is a new schema and execution candidate, not an old actor compatibility
adapter. There is no local flight model, PID fallback, hidden native property
access, 60Hz interpolation, or neural-network inference in this module.
"""
from copy import deepcopy
from types import SimpleNamespace
import math

import numpy as np

from ...competition.flight_control_ppo_delivery import OBSERVATION_SCHEMA,CONTROL_SCHEMA
from ...competition.flight_control_ppo_delivery.config import load_config,validate_config
from ...envs.JSBSim.core.catalog import Catalog as c
from ...envs.JSBSim.envs.singlecontrol_env import SingleControlEnv
from ...envs.JSBSim.tasks.heading_task import HeadingTask
from ...envs.JSBSim.utils.baiyepower_v2_physics import compute_energy_cost_terms
from ...envs.JSBSim.utils.flight_path_geometry import DirectionRateFilter,reference_direction
from ...scripts.experiments.c46_compact_observation import FEATURE_NAMES as ORIGINAL_FEATURE_NAMES

DT=.2
UID="A0100"
FEATURE_NAMES=list(ORIGINAL_FEATURE_NAMES)
FEATURE_NAMES[7]="equivalent_airspeed_over_340mps"
FEATURE_NAMES[31]="pilot_normal_load_up_over_5g"
FEATURE_NAMES[34]="pilot_load_valid"
FEATURE_NAMES=tuple(FEATURE_NAMES)
COMMAND_PATHS=("fcs/aileron-cmd-norm","fcs/elevator-cmd-norm","fcs/rudder-cmd-norm","fcs/throttle-cmd-norm")
SURFACE_PATHS=("fcs/left-aileron-pos-norm","fcs/right-aileron-pos-norm","fcs/elevator-pos-norm","fcs/rudder-pos-norm","fcs/throttle-pos-norm")
RAW_FIELDS=("simulation/sim-time-sec","simulation/dt","position/h-sl-ft","attitude/roll-rad","attitude/pitch-rad",
    "attitude/psi-deg","aero/alpha-deg","aero/beta-deg","velocities/p-rad_sec","velocities/q-rad_sec",
    "velocities/r-rad_sec","velocities/u-fps","velocities/v-fps","velocities/w-fps",
    "velocities/v-north-fps","velocities/v-east-fps","velocities/v-down-fps","velocities/ve-fps",
    "accelerations/n-pilot-z-norm")+COMMAND_PATHS+SURFACE_PATHS


def required_raw_fields():return list(RAW_FIELDS)


def _finite(value,name):
    if isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,float,np.integer,np.floating)) or not math.isfinite(value):
        raise ValueError("finite real value required: "+name)
    return float(value)


def plain(value):
    if isinstance(value,np.ndarray):return [plain(v) for v in value.tolist()]
    if isinstance(value,np.generic):return plain(value.item())
    if isinstance(value,dict):return {str(k):plain(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [plain(v) for v in value]
    if isinstance(value,float) and not math.isfinite(value):return None
    return value


def validate_raw(raw):
    if not isinstance(raw,dict):raise ValueError("raw own-aircraft property mapping required")
    out={key:_finite(raw[key],key) for key in RAW_FIELDS}
    if not 0.<out["simulation/dt"]<.1:raise ValueError("positive native simulation dt required")
    if out["velocities/ve-fps"]<=0:raise ValueError("positive measured EAS required")
    for path in COMMAND_PATHS[:3]:
        if not -1.<=out[path]<=1.:raise ValueError("native AER command outside physical [-1,1]")
    if not 0.<=out[COMMAND_PATHS[3]]<=1.:raise ValueError("native throttle command outside physical [0,1]")
    # A supplied alias is checked, never used to override canonical telemetry.
    for canonical,alias in (("attitude/roll-rad","attitude/phi-rad"),("attitude/pitch-rad","attitude/theta-rad")):
        if alias in raw and abs(_finite(raw[alias],alias)-out[canonical])>1e-7:
            raise ValueError("inconsistent actual attitude alias")
    return out


def reference(value):
    if not isinstance(value,dict):raise ValueError("explicit public reference mapping required")
    required=("heading_rad","height_m","ground_speed_mps","gamma_ref_rad","rates")
    if any(k not in value for k in required):raise ValueError("heading/height/groundspeed/gamma/rates reference required")
    out={k:_finite(value[k],k) for k in required if k!="rates"}
    rates=np.asarray(value["rates"],dtype=np.float64)
    if rates.shape!=(3,) or not np.isfinite(rates).all():raise ValueError("three finite mixed-unit target rates required")
    if out["ground_speed_mps"]<=0 or abs(out["gamma_ref_rad"])>=math.pi/2:
        raise ValueError("positive groundspeed and nonvertical reference required")
    out["heading_rad"]%=2*math.pi
    out["rates"]=rates.copy()
    preview=value.get("preview_gamma_rates_degps")
    if preview is None:out["preview_gamma_rates_degps"]=None
    else:
        if not isinstance(preview,(tuple,list,np.ndarray)) or len(preview)!=2:
            raise ValueError("published .2/.6s gamma-rate preview pair required")
        out["preview_gamma_rates_degps"]=tuple(None if v is None else _finite(v,"gamma-rate preview") for v in preview)
    return out


class RawStateFacade:
    """Only named actual raw reads, exact unit transforms and local task targets.

    No jsbsim_exec attribute or synthetic native state exists. Unknown property
    access raises even if a legacy caller expected a permissive native getter.
    """
    def __init__(self,raw,ref,logical_time_sec):self.update(raw,ref,logical_time_sec)

    def update(self,raw,ref,logical_time_sec):
        self.raw=deepcopy(raw);self.reference=ref;self.logical_time_sec=logical_time_sec
        self.local={c.target_heading_deg.name_jsbsim:math.degrees(ref["heading_rad"]),
            c.target_altitude_ft.name_jsbsim:ref["height_m"]/.3048,
            c.baiyepower_target_speed_mps.name_jsbsim:ref["ground_speed_mps"]}

    def get_sim_time(self):return self.logical_time_sec

    def get_property_value(self,prop):
        name=prop if isinstance(prop,str) else prop.name_jsbsim
        if name=="simulation/sim-time-sec":return self.logical_time_sec
        if name in self.local:return self.local[name]
        if name in self.raw:return self.raw[name]
        simple={"position/h-sl-m":("position/h-sl-ft",.3048),
            "attitude/phi-rad":("attitude/roll-rad",1.),"attitude/theta-rad":("attitude/pitch-rad",1.),
            "attitude/phi-deg":("attitude/roll-rad",180./math.pi),
            "attitude/theta-deg":("attitude/pitch-rad",180./math.pi),
            "attitude/psi-rad":("attitude/psi-deg",math.pi/180.),
            "velocities/h-dot-fps":("velocities/v-down-fps",-1.)}
        for axis in ("u","v","w","v-north","v-east","v-down"):
            simple["velocities/"+axis+"-mps"]=("velocities/"+axis+"-fps",.3048)
        if name in simple:
            path,scale=simple[name];return self.raw[path]*scale
        raise KeyError("unavailable platform property; no fabricated native value: "+name)

    def get_property_values(self,props):return [self.get_property_value(p) for p in props]

    def set_property_value(self,prop,value):
        name=prop if isinstance(prop,str) else prop.name_jsbsim
        if name not in self.local:raise ValueError("facade cannot write platform or unregistered native properties: "+name)
        self.local[name]=_finite(value,name)


class _Task(HeadingTask):
    action_type="continuous"
    obs_version="v5_target_rates"
    action_var=(c.fcs_aileron_cmd_norm,c.fcs_elevator_cmd_norm,c.fcs_rudder_cmd_norm,c.fcs_throttle_cmd_norm)
    _throttle_command_bounds=HeadingTask._throttle_command_bounds
    action_to_policy_coordinates=HeadingTask.action_to_policy_coordinates
    normalize_action=HeadingTask.normalize_action
    def __init__(self,config,env):self.config=config;self.env=env
    @staticmethod
    def get_current_speed_mps(sim):
        return math.sqrt(sum(float(sim.get_property_value(p))**2 for p in (c.velocities_u_mps,c.velocities_v_mps,c.velocities_w_mps)))
    def get_target_speed_mps(self,sim):return sim.reference["ground_speed_mps"]
    def get_target_rate_commands(self,env,agent_id):return env.agents[agent_id].reference["rates"].copy()
    def get_speed_error_mps(self,sim):return self.get_target_speed_mps(sim)-self.get_current_speed_mps(sim)
    def get_baiyepower_v2_transition_context(self,agent_id):
        sim=self.env.agents[agent_id]
        terms=compute_energy_cost_terms(altitude_m=sim.get_property_value(c.position_h_sl_m),
            target_altitude_m=sim.reference["height_m"],u_mps=sim.get_property_value(c.velocities_u_mps),
            v_mps=sim.get_property_value(c.velocities_v_mps),w_mps=sim.get_property_value(c.velocities_w_mps),
            target_speed_mps=sim.reference["ground_speed_mps"])
        return SimpleNamespace(pre_energy_terms=terms,target_speed_mps=sim.reference["ground_speed_mps"],
            target_altitude_m=sim.reference["height_m"],target_heading_rad=sim.reference["heading_rad"])


class _NominalContext(SingleControlEnv):
    """Reuse frozen computation methods without invoking any Env constructor."""
    def __init__(self,cfg,facade):
        self.config=SimpleNamespace(**deepcopy(cfg))
        self._jsbsims={UID:facade};self.task=_Task(self.config,self)
        self.current_step=0;self.sim_freq=60;self.agent_interaction_steps=12
        self._reset_controller_measurement_noise()
        self._reset_coordinated_residual()
        self._reset_action_projector()


class DeliveryController:
    def __init__(self,config=None):
        self.configuration=load_config() if config is None else deepcopy(validate_config(config))
        self._context=None;self._pending=False;self._faulted=False;self._observations=0;self._actions=0

    def reset(self,raw,public_reference,logical_time_sec=0.):
        if _finite(logical_time_sec,"reset logical time")!=0.:raise ValueError("explicit episode reset at logical time0 required")
        actual=validate_raw(raw);ref=reference(public_reference)
        facade=RawStateFacade(actual,ref,0.)
        initial=facade.get_property_values(_Task.action_var)
        cfg=self.configuration["nominal_config"]
        if any(not lo<=v<=hi for v,lo,hi in zip(initial,cfg["ActionProjector_action_min"],cfg["ActionProjector_action_max"])):
            raise ValueError("actual initial command outside registered QP box; no silent reset clipping")
        self._context=_NominalContext(cfg,facade)
        self._filter=DirectionRateFilter(float(self.configuration["direction_filter_tau_sec"]))
        self._filter_ready=False;self._faulted=False;self._pending=False
        self._observations=0;self._actions=0;self._last_raw_time=None;self._logical_time=None
        self._last_result=None
        return self._observe(actual,ref,0.,initial=True)

    def observe(self,raw,public_reference,logical_time_sec):
        if self._context is None or self._faulted:raise RuntimeError("explicit reset required before platform observation")
        if self._pending:raise RuntimeError("each observed policy input permits exactly one action before next observation")
        try:return self._observe(validate_raw(raw),reference(public_reference),_finite(logical_time_sec,"logical time"),initial=False)
        except Exception:self._faulted=True;raise

    def _observe(self,raw,ref,logical_time,initial):
        if not initial and (abs(logical_time-self._logical_time-DT)>1e-9 or raw["simulation/sim-time-sec"]<=self._last_raw_time):
            raise ValueError("strict 5Hz logical cadence and increasing actual clock required")
        self._context.agents[UID].update(raw,ref,logical_time)
        self._context.current_step=self._actions+1
        self._raw=raw;self._reference=ref
        self._logical_time=logical_time;self._last_raw_time=raw["simulation/sim-time-sec"]
        ned=np.array([raw["velocities/v-"+a+"-fps"]*.3048 for a in ("north","east","down")],dtype=np.float64)
        ground=float(np.linalg.norm(ned));horizontal=float(np.hypot(ned[0],ned[1]))
        if ground<=1. or horizontal<=1e-6:raise ValueError("nonvertical valid flight direction required; no placeholder direction")
        direction=ned/ground
        desired=reference_direction(ref["heading_rad"],ref["gamma_ref_rad"])
        if not self._filter_ready:
            actual_rate,ref_rate=self._filter.reset(direction,desired);self._filter_ready=True
        else:actual_rate,ref_rate=self._filter.update(direction,desired,DT)
        self._actual_rate=actual_rate;self._reference_rate=ref_rate
        gamma=math.atan2(-ned[2],horizontal);course=math.atan2(ned[1],ned[0])
        heading=ref["heading_rad"];gref=ref["gamma_ref_rad"]
        tangent=np.array([-math.sin(gref)*math.cos(heading),-math.sin(gref)*math.sin(heading),-math.cos(gref)])
        rate_error=float(np.dot(actual_rate-ref_rate,tangent))
        env=self._context;history=env._action_projector.get_observation_state(UID)
        prev=env.task.action_to_policy_coordinates(history["prev"])
        prev2=env.task.action_to_policy_coordinates(history["prev2"])
        roll,pitch=raw["attitude/roll-rad"],raw["attitude/pitch-rad"]
        obs=[(ref["height_m"]-raw["position/h-sl-ft"]*.3048)/100.,(ref["ground_speed_mps"]-ground)/10.,
            math.remainder(heading-course,2*math.pi)/math.pi,(gref-gamma)/math.radians(10.),
            gamma/(math.pi/2),raw["position/h-sl-ft"]*.3048/10000.,ground/340.,raw["velocities/ve-fps"]*.3048/340.,
            math.sin(roll),math.cos(roll),math.sin(pitch),math.cos(pitch),
            raw["velocities/p-rad_sec"],raw["velocities/q-rad_sec"],raw["velocities/r-rad_sec"],
            raw["aero/alpha-deg"]/15.,1.,1.]+list(prev)+list(prev2)
        obs += [raw[p] for p in SURFACE_PATHS]
        obs += [-raw["accelerations/n-pilot-z-norm"]/5.,1.,1.,1.,rate_error/math.radians(3.)]
        preview=ref["preview_gamma_rates_degps"]
        if preview is None:obs += [0.,0.,0.,0.]
        else:
            for value in preview:obs += [0.,0.] if value is None else [value/10.,1.]
        result=np.asarray(obs,dtype=np.float32)
        if result.shape!=(40,) or not np.isfinite(result).all():raise ValueError("finite40D platform observation required")
        self._observation=result;self._pending=True;self._observations+=1
        return result.copy()

    def apply(self,residual):
        if self._context is None or self._faulted or not self._pending:raise RuntimeError("one fresh observation required per control action")
        try:
            action=np.asarray(residual,dtype=np.float32).reshape(-1)
            if action.shape!=(4,) or not np.isfinite(action).all() or np.any(np.abs(action)>1.):
                raise ValueError("four bounded finite actual PPO residuals required; no extra action clipping")
            env=self._context
            composed=env._apply_coordinated_residual(UID,action)
            physical=env.task.normalize_action(env,UID,composed)
            command=np.asarray(env._apply_action_projector(UID,physical),dtype=np.float32)
            qp=env._action_projector.get_last_log(UID)
            if command.shape!=(4,) or not np.isfinite(command).all() or qp["qp_projector_fail_streak"]!=0:
                raise ValueError("invalid original QP output; no fallback control permitted")
            result=dict(schema=CONTROL_SCHEMA,observation_schema=OBSERVATION_SCHEMA,
                obs40=self._observation.copy(),residual_action=action.copy(),command=command.copy(),
                composed_policy_action=np.asarray(composed).copy(),physical_proposal=np.asarray(physical).copy(),
                composition_log=plain(env.get_coordinated_residual_log(UID)),qp_log=plain(qp),
                nominal_state=plain(env._energy_trim_states),reference=plain(self._reference),
                logical_time_sec=self._logical_time,actual_raw_time_sec=self._last_raw_time,
                execution="QP_output_5Hz_ZOH",smoother_used=False,PID_fallback=False,
                physical_residual_authority=[.1,.05,.05,.22],actor_forward_evaluations=0)
            self._pending=False;self._actions+=1;self._last_result=result
            return result
        except Exception:self._faulted=True;raise

    def observation(self):
        if self._context is None:raise RuntimeError("reset required")
        return self._observation.copy()

    def snapshot(self):
        if self._context is None:return dict(initialized=False)
        return dict(initialized=True,schema=CONTROL_SCHEMA,observations=self._observations,actions=self._actions,
            pending_action=self._pending,faulted=self._faulted,logical_time_sec=self._logical_time,
            actual_raw_time_sec=self._last_raw_time,nominal_state=plain(self._context._energy_trim_states),
            qp_history=plain(self._context._action_projector.get_observation_state(UID)),
            actual_direction_rate=plain(self._actual_rate),reference_direction_rate=plain(self._reference_rate))
