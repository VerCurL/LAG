"""Versioned, local delivery configuration; no historical artifact dependency."""
from copy import deepcopy
import json
import math
from pathlib import Path

DEFAULT_CONFIG = Path(__file__).with_name("default_config.json")


def load_config(path=None):
    value=json.loads((DEFAULT_CONFIG if path is None else Path(path)).read_text(encoding="utf-8"))
    validate_config(value)
    return deepcopy(value)


def validate_config(value):
    if not isinstance(value,dict) or value.get("schema")!="platform-ppo-delivery-config-v1":
        raise ValueError("explicit platform delivery configuration schema required")
    cfg=value.get("nominal_config")
    if not isinstance(cfg,dict):raise ValueError("registered nominal_config required")
    required={"ActionThrottle_physical_min":0.,"ActionThrottle_physical_max":1.,
        "CoordinatedResidual_policy_throttle_center":.5,
        "CoordinatedResidual_policy_throttle_half_range":.5,
        "CoordinatedResidual_min_adaptive_trim_throttle":0.,
        "CoordinatedResidual_max_adaptive_trim_throttle":1.,
        "CoordinatedResidual_mode":"actuator_residual_fixed_center",
        "CoordinatedResidual_enable":True,"ActionProjector_enable":True,
        "ActionSmoother_enable":False,"ControllerMeasurementNoise_enable":False}
    if any(cfg.get(k)!=v for k,v in required.items()):
        raise ValueError("platform action mapping/nominal/QP/ZOH contract changed")
    if cfg.get("ActionProjector_action_min")!=[-1.,-1.,-1.,0.] or cfg.get("ActionProjector_action_max")!=[1.,1.,1.,1.]:
        raise ValueError("platform physical QP bounds must be AER[-1,1], throttle[0,.9]")
    expected=[.1,.05,.05,.22/.5]
    if cfg.get("CoordinatedResidual_residual_scale")!=expected:
        raise ValueError("registered physical residual authority must remain .1/.05/.05/.22")
    trim=cfg.get("CoordinatedResidual_trim_physical_throttle")
    if isinstance(trim,bool) or not isinstance(trim,(int,float)) or not math.isfinite(trim) or not 0.<=trim<=1.:
        raise ValueError("finite independent trim command inside physical box required")
    if value.get("execution")!="5Hz_QP_output_zero_order_hold_no_external_60Hz_smoothing":
        raise ValueError("training and deployment must share the 5Hz ZOH execution")
    return value
