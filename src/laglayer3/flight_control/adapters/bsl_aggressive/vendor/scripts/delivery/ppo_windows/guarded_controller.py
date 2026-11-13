"""Isolated post-training control adapter for frozen C83 platform PPO weights.

The original policy observation, nominal controller, coordinate map, QP and
5Hz execution are retained. Four actual Actor outputs remain necessary. This
adapter is a distribution change requiring its own frozen-weight validation;
it is neither fresh training nor evidence that the original C83 passed.
"""
from copy import deepcopy

import numpy as np

from ....competition.flight_control_ppo_delivery import CONTROL_SCHEMA as PARENT_CONTROL_SCHEMA,OBSERVATION_SCHEMA
from ....competition.flight_control_ppo_delivery.controller import DeliveryController,UID,plain


SCHEMA="platform-residual-PPO-control-adapter-v1"
RAW_RESIDUAL_AUTHORITY=(.1,.05,.05,.22)
EFFECTIVE_RESIDUAL_MULTIPLIERS=(.2,1.,1.,.01/.22)
PHYSICAL_RESIDUAL_AUTHORITY=(.02,.05,.05,.01)
ELEVATOR_TRIM_FEEDFORWARD=-.18599601816358236


def adapter_config():
    """Pure literal configuration for registration and independent replay."""
    return dict(schema=SCHEMA,parent_control_schema=PARENT_CONTROL_SCHEMA,observation_schema=OBSERVATION_SCHEMA,
        raw_residual_authority=list(RAW_RESIDUAL_AUTHORITY),
        effective_residual_multipliers=list(EFFECTIVE_RESIDUAL_MULTIPLIERS),
        physical_residual_authority=list(PHYSICAL_RESIDUAL_AUTHORITY),
        elevator_trim_feedforward=ELEVATOR_TRIM_FEEDFORWARD,
        feedforward_source=dict(experiment="C50-E2",artifact="model.json",field="trim.commands[1]",
            command_basis="competition-native-aert-zero-hidden-trim-v1",hidden_trim_written=False,
            scope="independent local sendable equilibrium center; no startup or envelope guarantee"),
        arithmetic="effective=float32(raw*float32(multiplier)); normalize original composition; elevator=float32(float64(pre_E)+trim); one original QP",
        weights_trained_with_adapter=False,weight_tensors_modified=False,
        original_C83_performance_inherited=False,adapter_validation_required=True,
        PPO_enabled=True,PID_fallback=False,QP_calls_per_action=1,smoother_used=False,
        initialization_or_public_reference_changed=False)


class GuardedController(DeliveryController):
    """One original composition and one original QP per genuine Actor action.

    Parent reset/observe builds the unchanged40D input and keeps real history.
    Parent apply is deliberately not called: its QP would otherwise execute
    before the feedforward, changing both the operation and history twice.
    """
    def apply(self,residual):
        if self._context is None or self._faulted or not self._pending:
            raise RuntimeError("one fresh observation required per control action")
        try:
            raw=np.asarray(residual,dtype=np.float32).reshape(-1)
            if raw.shape!=(4,) or not np.isfinite(raw).all() or np.any(np.abs(raw)>1.):
                raise ValueError("four bounded finite actual PPO residuals required; no clipping or fallback")
            raw=raw.copy()
            effective=(raw*np.asarray(EFFECTIVE_RESIDUAL_MULTIPLIERS,dtype=np.float32)).astype(np.float32)
            env=self._context
            composed=env._apply_coordinated_residual(UID,effective)
            pre=np.asarray(env.task.normalize_action(env,UID,composed),dtype=np.float32).copy()
            if pre.shape!=(4,) or not np.isfinite(pre).all():
                raise ValueError("original normalized proposal must have four finite physical channels")
            physical=pre.copy()
            physical[1]=np.float32(float(pre[1])+ELEVATOR_TRIM_FEEDFORWARD)
            # High-level accelerate is an explicit power request in this fork.
            # Project it once with the original slew/box constraints and keep
            # the actual command in QP history (never override after projection).
            aggressive_accelerate = bool(getattr(self, "_aggressive_accelerate", False))
            assault_power = bool(getattr(self, "_assault_power", False))
            if assault_power:
                physical[3] = np.float32(1.0)
            elif aggressive_accelerate:
                physical[3] = np.float32(.9)
            physical[3]=np.float32(min(float(physical[3]),getattr(self,'_speed_throttle_ceiling',1.)))
            # There is no second normalize, pre-QP clip, smoother or parent.apply.
            command=np.asarray(env._apply_action_projector(UID,physical),dtype=np.float32)
            qp=env._action_projector.get_last_log(UID)
            if command.shape!=(4,) or not np.isfinite(command).all() or qp["qp_projector_fail_streak"]!=0:
                raise ValueError("invalid original QP output; no fallback control permitted")
            composition=plain(env.get_coordinated_residual_log(UID))
            result=dict(schema=SCHEMA,parent_control_schema=PARENT_CONTROL_SCHEMA,observation_schema=OBSERVATION_SCHEMA,
                obs40=self._observation.copy(),residual_action=raw.copy(),raw_residual_action=raw.copy(),
                effective_residual_action=effective.copy(),command=command.copy(),
                composed_policy_action=np.asarray(composed).copy(),
                nominal_physical_proposal=pre.copy(),physical_proposal=physical.copy(),
                requested_physical_feedforward=[0.,ELEVATOR_TRIM_FEEDFORWARD,0.,0.],
                actual_physical_feedforward_delta=(physical-pre).copy(),
                composition_log=deepcopy(composition),qp_log=plain(qp),
                nominal_state=plain(env._energy_trim_states),reference=plain(self._reference),
                logical_time_sec=self._logical_time,actual_raw_time_sec=self._last_raw_time,
                execution="QP_output_5Hz_ZOH",smoother_used=False,PID_fallback=False,
                physical_residual_authority=list(PHYSICAL_RESIDUAL_AUTHORITY),actor_forward_evaluations=0,
                adapter=adapter_config(),
                original_composition_record=dict(parent_control_schema=PARENT_CONTROL_SCHEMA,
                    residual_action=effective.copy(),composed_policy_action=np.asarray(composed).copy(),
                    physical_proposal=pre.copy(),composition_log=deepcopy(composition),
                    scope="original compose/normalize record only; unadapted command was not projected or flown"))
            self._pending=False;self._actions+=1;self._last_result=result
            result["aggressive_accelerate"] = aggressive_accelerate
            result["assault_power"] = assault_power
            return result
        except Exception:
            self._faulted=True;raise

    def snapshot(self):
        result=super().snapshot()
        if result["initialized"]:
            result["parent_control_schema"]=result["schema"]
            result["schema"]=SCHEMA
        result["adapter"]=adapter_config()
        return result
