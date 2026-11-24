"""Optional deterministic Actor-only wrapper around the shared controller."""
from pathlib import Path
import hashlib

import numpy as np
import torch

from ...competition.flight_control_ppo_delivery import OBSERVATION_SCHEMA
from ...competition.flight_control_ppo_delivery.controller import DeliveryController
from ...competition.flight_control_ppo_delivery.policy import SCHEMA,create_platform_policy,make_compact_args


def load_policy(path,expected_sha256):
    """Load only a bound fresh-platform endpoint; never accepts legacy40D weights."""
    path=Path(path)
    if not isinstance(expected_sha256,str) or len(expected_sha256)!=64:
        raise ValueError("explicit registered platform checkpoint SHA-256 required")
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    if digest!=expected_sha256:raise ValueError("registered platform PPO checkpoint changed")
    payload=torch.load(path,map_location="cpu",weights_only=True)
    metadata=payload.get("metadata",{}) if isinstance(payload,dict) else {}
    if metadata.get("schema")!=SCHEMA or metadata.get("observation_schema")!=OBSERVATION_SCHEMA:
        raise ValueError("new platform40D weights required; legacy40D weights are incompatible")
    args=make_compact_args(gain=metadata["action_gain"])
    policy=create_platform_policy(metadata["initialization_seed"],args=args)
    policy.load_checkpoint_payload(payload);policy.prep_rollout()
    return policy


class ActorDelivery:
    """One actual GRU history per aircraft; no PID fallback or probing action."""
    def __init__(self,policy,controller=None):
        metadata=policy.metadata()
        if metadata.get("schema")!=SCHEMA or metadata.get("observation_schema")!=OBSERVATION_SCHEMA:
            raise ValueError("explicit fresh-platform policy identity required")
        self.policy=policy
        self.controller=DeliveryController() if controller is None else controller
        self.actor_forward_evaluations=0;self.hidden=None;self.mask=None

    def reset(self,raw,reference,logical_time_sec=0.):
        obs=self.controller.reset(raw,reference,logical_time_sec)
        self.hidden=np.zeros((1,1,64),dtype=np.float32)
        self.mask=np.zeros((1,1),dtype=np.float32)
        self.actor_forward_evaluations=0
        return obs

    def observe(self,raw,reference,logical_time_sec):
        return self.controller.observe(raw,reference,logical_time_sec)

    def act(self):
        if self.hidden is None:raise RuntimeError("explicit aircraft episode reset required")
        with torch.no_grad():
            action,post=self.policy.act(self.controller.observation()[None,:],self.hidden,self.mask,deterministic=True)
        self.actor_forward_evaluations+=1
        actual=action.detach().cpu().numpy().reshape(4)
        result=self.controller.apply(actual)
        self.hidden=post.detach().cpu().numpy().copy();self.mask.fill(1.)
        return result
