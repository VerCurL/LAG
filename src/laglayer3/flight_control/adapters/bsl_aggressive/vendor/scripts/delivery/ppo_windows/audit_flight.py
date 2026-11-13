"""Read-only raw-packet/Actor/GRU/control replay of a portable Windows flight.

Expected observations/actions are recomputed from response packets and bound
weights, never taken from producer policy logs. The original control module is
shared deliberately: this verifies actual runtime execution, not an independent
mathematical proof of that controller. No FDM/socket/platform call is made.
"""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import struct
import sys


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"),
        parse_constant=lambda v:(_ for _ in ()).throw(ValueError("nonfinite JSON "+v)))


def lines(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(v,parse_constant=lambda x:(_ for _ in ()).throw(ValueError(x))) for v in stream if v.strip()]


def packet(path):
    raw=Path(path).read_bytes()
    if len(raw)<4 or struct.unpack("!I",raw[:4])[0]!=len(raw)-4 or len(raw)>16*1024*1024+4:
        raise ValueError("actual packet length header/payload mismatch")
    return json.loads(gzip.decompress(raw[4:]).decode("utf-8"),
        parse_constant=lambda v:(_ for _ in ()).throw(ValueError(v)))


def same(actual,expected,label,tolerance=1e-5):
    """Exact structure/booleans and finite numeric agreement, with max error."""
    if isinstance(expected,dict):
        if not isinstance(actual,dict) or set(actual)!=set(expected):raise ValueError(label+": mapping keys differ")
        return max((same(actual[k],v,label+"."+k,tolerance) for k,v in expected.items()),default=0.)
    if isinstance(expected,(list,tuple)):
        if not isinstance(actual,(list,tuple)) or len(actual)!=len(expected):raise ValueError(label+": sequence shape differs")
        return max((same(a,b,label,tolerance) for a,b in zip(actual,expected)),default=0.)
    if isinstance(expected,bool) or expected is None or isinstance(expected,str):
        if type(actual)!=type(expected) or actual!=expected:raise ValueError(label+": identity value differs")
        return 0.
    if isinstance(expected,(int,float)):
        if isinstance(actual,bool) or not isinstance(actual,(int,float)) or not math.isfinite(actual) or not math.isfinite(expected):
            raise ValueError(label+": finite numeric value required")
        delta=abs(actual-expected)
        if delta>tolerance:raise ValueError(label+": numeric replay differs by "+str(delta))
        return delta
    raise TypeError("unsupported audit field: "+label)


def independent_reference(profile,index,heading,height,ground_speed=160.02):
    if profile not in ("hold","heading30","height100") or type(index) is not int or not 0<=index<600:
        raise ValueError("unregistered flight profile/index")
    shift=30. if 50<=index<200 else (-30. if 200<=index<350 else 0.)
    delta_height=(100. if 50<=index<200 else (-100. if 200<=index<350 else 0.)) if profile=="height100" else 0.
    return dict(heading_rad=(heading+(math.radians(shift) if profile=="heading30" else 0.))%(2*math.pi),
        height_m=height+delta_height,ground_speed_mps=ground_speed,gamma_ref_rad=0.,rates=[0.,0.,0.],preview_gamma_rates_degps=None)


def _f32(value):return struct.unpack("<f",struct.pack("<f",value))[0]


def clock_receipt(observation,index,previous=None):
    expected=_f32(index*12*_f32(1/60.))
    result=dict(index=index,logical_control_time_sec=index*.2,expected_native_time_binary32=expected,
        native_steps_per_action=12,controller_dt_sec=.2,physical_observations_changed=False,aircraft={})
    for side in ("red","blue"):
        raw=observation[side][side+"_0"];time,dt=raw["simulation/sim-time-sec"],raw["simulation/dt"]
        if (isinstance(time,bool) or isinstance(dt,bool) or not math.isfinite(time) or not math.isfinite(dt)
                or _f32(time)!=expected or _f32(dt)!=_f32(1/60.)
                or previous is not None and time<=previous[side][side+"_0"]["simulation/sim-time-sec"]):
            raise ValueError("actual native binary32 clock/grid/ordering differs")
        result["aircraft"][side]=dict(raw_time_sec=time,raw_dt_sec=dt,raw_time_binary32=_f32(time),logical_control_time_sec=index*.2)
    return result


def safe_frame(observation,contract):
    for side in ("red","blue"):
        raw=observation[side][side+"_0"]
        if (type(raw["LifeCurrent"]) not in (int,float) or not math.isfinite(raw["LifeCurrent"])
                or raw["LifeCurrent"]<=0 or type(raw["DeathEvent"]) not in (int,float) or raw["DeathEvent"]!=99):return False
        values=dict(height_m=raw["position/h-sl-ft"]*.3048,eas_mps=raw["velocities/ve-fps"]*.3048,
            alpha_deg=raw["aero/alpha-deg"],beta_deg=raw["aero/beta-deg"],roll_deg=math.degrees(raw["attitude/roll-rad"]),pitch_deg=math.degrees(raw["attitude/pitch-rad"]))
        if any(not math.isfinite(values[k]) or not a<=values[k]<=b for k,(a,b) in contract["observed_domain"].items()):return False
    return True


def raw_identity(output):
    excluded={"registration.json","report.json","independent_audit.json","independent_audit_failure.json","completion.json"}
    return {p.relative_to(output).as_posix():sha(p) for p in output.rglob("*") if p.is_file() and p.name not in excluded}


def _wire(output,frames,decisions,initial_request,failed):
    count=len(frames);names={p.name for p in (output/"wire").glob("*.bin")}
    required={"%04d.%s.bin"%(i,kind) for i in range(count) for kind in ("request","response")}
    extras=names-required
    if not required<=names or not extras<={"%04d.request.bin"%count,"%04d.response.bin"%count} or extras and not failed:
        raise ValueError("actual packet inventory is missing or includes unexplained traffic")
    if "%04d.response.bin"%count in extras and "%04d.request.bin"%count not in extras:
        raise ValueError("orphan response without sent request")
    for i,frame in enumerate(frames):
        if frame.get("index")!=i:raise ValueError("actual frame sequence differs")
        same(frame["observation"],packet(output/"wire"/("%04d.response.bin"%i)),"actual response/frame",0.)
        request=initial_request if i==0 else decisions[i-1]["request"]
        same(packet(output/"wire"/("%04d.request.bin"%i)),request,"actual sent request",0.)
    if "%04d.request.bin"%count in extras:
        if len(decisions)!=count:raise ValueError("failed-exchange packet requires actual prepared decision")
        same(packet(output/"wire"/("%04d.request.bin"%count)),decisions[-1]["request"],"failed sent request",0.)
    return sorted(extras)


def replay(package,registration,frames,decisions,report):
    """Independent driver: shared nominal/QP, direct actual Actor invocation."""
    import numpy as np
    import torch
    from ....competition.flight_control_ppo_delivery.actor import load_policy
    from ....competition.flight_control_ppo_delivery.controller import plain
    from ....competition.flight_control_eas import runtime,evaluation
    from ....competition.flight_control_eas_development.pid_speed_proportional import make_controller
    from ....scripts.delivery.ppo_windows.inference import load_manifest,configure_inference_runtime,require_cpu_policy,controller_from_manifest,controller_identity
    actual_runtime=configure_inference_runtime()
    same(registration["inference_runtime"],actual_runtime,"registered CPU/thread1 inference",0.)
    manifest=load_manifest(package)
    same(registration["controller_identity"],controller_identity(manifest),"manifest-selected control adapter",0.)
    policy=load_policy(package/manifest["checkpoint"],manifest["checkpoint_sha256"])
    require_cpu_policy(policy)
    ctrl=controller_from_manifest(package,manifest)
    blue=make_controller("pid",read(package/"assets/pid_model.json"));blue.reset()
    hidden=np.zeros((1,1,64),dtype=np.float32);mask=np.zeros((1,1),dtype=np.float32)
    bases={s:dict(heading=math.radians(frames[0]["observation"][s][s+"_0"]["attitude/psi-deg"]),
        height=frames[0]["observation"][s][s+"_0"]["position/h-sl-ft"]*.3048) for s in ("red","blue")}
    maximum=0.
    for i,row in enumerate(decisions):
        if row.get("index")!=i or i>=len(frames):raise ValueError("policy decision has no corresponding actual observation")
        actual=frames[i]["observation"]
        refs={s:independent_reference(registration["profile"],i,**bases[s]) for s in bases}
        same(row["reference"],refs,"public reference",0.)
        obs=ctrl.reset(actual["red"]["red_0"],refs["red"],0.) if i==0 else ctrl.observe(actual["red"]["red_0"],refs["red"],i*.2)
        before=hidden.copy();mask_before=mask.copy()
        with torch.no_grad():action,post=policy.act(obs[None,:],before,mask_before,deterministic=True)
        expected=ctrl.apply(action.detach().cpu().numpy().reshape(4))
        hidden=post.detach().cpu().numpy().copy();mask.fill(1.)
        expected.update(actor_state_before=before,actor_state_post=hidden.copy(),mask_before=mask_before,
            deterministic=True,actor_forward_evaluations=i+1,critic_forward_evaluations=0,PID_fallback=False)
        maximum=max(maximum,same(row["red_actual_PPO"],plain(expected),"actual red PPO/hidden/controller replay"))
        state=runtime.parse_aircraft(actual["blue"]["blue_0"])
        from dataclasses import replace
        logical=replace(state,time_sec=i*.2)
        ground=math.sqrt(state.v_north_mps**2+state.v_east_mps**2+state.v_down_mps**2)
        ref=runtime.make_reference(dict(heading_deg=math.degrees(refs["blue"]["heading_rad"]),
            height_m=refs["blue"]["height_m"],eas_mps=160.02*state.eas_mps/ground))
        blue_output,inner=blue.step(logical,ref,.2)
        maximum=max(maximum,same(row["blue_PID_output"],plain(blue_output.__dict__),"independent blue PID replay"),
            same(row["blue_inner_reference"],plain(inner.__dict__),"blue reference replay"))
        red_action=dict(zip(("fcs/aileron-cmd-norm","fcs/elevator-cmd-norm","fcs/rudder-cmd-norm","fcs/throttle-cmd-norm"),
            map(float,expected["command"])),mode=0,**{"fcs/weapon-launch":0,"switch-missile":0,"change-target":9})
        request=dict(red=dict(red_0=red_action),blue=dict(blue_0=evaluation.complete_action(blue_output)))
        maximum=max(maximum,same(row["request"],request,"actually sent PPO/PID command"))
    if report["actual_actor_forward_evaluations"]!=len(decisions):
        raise ValueError("Actor count differs from fully reconstructible decision evidence")
    return dict(independent_actor_replay_evaluations=len(decisions),critic_forward_evaluations=0,
        maximum_replay_abs_difference=maximum,producer_Actor_wrapper_used_as_oracle=False,
        source_of_expected_policy_inputs="actual response packets and independently reconstructed public task",
        shared_original_nominal_QP_controller=True,independent_mathematical_controller_proof=False)


def audit(package,output,save=True):
    package,output=Path(package).resolve(),Path(output).resolve()
    from ....scripts.delivery.ppo_windows.inference import load_manifest
    from ....competition.flight_control_eas.platform_reset_v1 import validate_initial,DEFAULT_TOLERANCES
    manifest=load_manifest(package);reg=read(output/"registration.json");report=read(output/"report.json")
    if (reg["schema"]!="portable-platform-PPO-demo-v1" or report["schema"]!=reg["schema"]
            or reg["package_manifest_sha256"]!=sha(package/"delivery_manifest.json")
            or reg["checkpoint_sha256"]!=manifest["checkpoint_sha256"]
            or report.get("registration_sha256")!=sha(output/"registration.json")
            or reg.get("red_actual_Actor") is not True or reg.get("blue_independent_PID") is not True
            or reg.get("PID_fallback") is not False or reg.get("duration_sec")!=120.
            or reg.get("expected_frames")!=601 or reg.get("expected_joint_actions")!=600):
        raise ValueError("registered portable PPO endpoint/flight identity differs")
    identity=raw_identity(output)
    if identity!=report["raw_identity"]:raise ValueError("actual raw flight inventory changed after report")
    frames,decisions,receipts,pacing=[lines(output/n) for n in ("frames.jsonl","decisions.jsonl","clock_receipts.jsonl","pacing.jsonl")]
    if report["frames"]!=len(frames) or not frames or len(decisions)>len(frames):
        raise ValueError("observed frame/decision budget differs")
    initial=read(package/"assets/platform_initial.json");initial_request=read(package/"assets/platform_init_request.json")
    contract=read(package/"assets/platform_contract.json")
    orphan=_wire(output,frames,decisions,initial_request,report["error"] is not None)
    initial_check=validate_initial(frames[0]["observation"],initial,DEFAULT_TOLERANCES)
    if (output/"initial_signature.json").exists():same(read(output/"initial_signature.json"),initial_check,"actual initial signature")
    if not initial_check["passed"] and decisions:raise ValueError("commands were applied after rejected initialization")
    unsafe=[]
    for i,row in enumerate(frames):
        expected=clock_receipt(row["observation"],i,None if i==0 else frames[i-1]["observation"])
        if i<len(receipts):same(receipts[i],expected,"native clock receipt",0.)
        if not safe_frame(row["observation"],contract):unsafe.append(i)
    if len(receipts)!=len(frames) and not (report["error"] is not None and len(receipts)==len(frames)-1 and unsafe==[len(frames)-1]):
        raise ValueError("native clock receipts do not match actual checked prefix")
    if unsafe and (unsafe!=[len(frames)-1] or len(decisions)>unsafe[0] or report["error"] is None):
        raise ValueError("unsafe observation did not stop the real command stream")
    if len(pacing)!=len(frames)-1 or report["joint_actions"]!=len(pacing):raise ValueError("actual pacing/action count differs")
    previous_receive=0.
    for i,row in enumerate(pacing,1):
        if row["index"]!=i or abs(row["scheduled_offset_sec"]-i*.2)>1e-10:
            raise ValueError("wall-clock paced command schedule differs")
        sent,received=row["send_offset_sec"],row["receive_offset_sec"]
        if not math.isfinite(sent) or not math.isfinite(received) or sent<i*.2-1e-6 or sent<previous_receive or received<sent:
            raise ValueError("commands were sent early or out of synchronous pacing order")
        previous_receive=received
    reconstructed=replay(package,reg,frames,decisions,report)
    complete=report["error"] is None and len(frames)==601 and len(decisions)==600
    if type(report.get("complete")) is not bool or report["complete"]!=complete:
        raise ValueError("complete120s identity differs from actual raw exposure")
    process=report["launched_process"]
    if complete and (not initial_check["passed"] or unsafe or report.get("engineering_pass") is not True
            or report["cleanup"] not in ("owned_process_stopped","already_exited") or process.get("identity_verified") is not True
            or type(process.get("pid")) is not int or process["pid"]<=0
            or not isinstance(process.get("creation"),str) or not process["creation"]
            or Path(process["executable"]).resolve()!=Path(reg["executable"]).resolve()
            or report.get("runtime_unchanged") is not True or report.get("executable_unchanged") is not True):
        raise ValueError("completed PPO flight lacks valid initialization/owned process/cleanup")
    load_manifest(package)
    result=dict(schema="portable-platform-PPO-runtime-replay-audit-v1",audit_pass=True,flight_complete=complete,
        engineering_pass=bool(complete and report["engineering_pass"]),registration_sha256=sha(output/"registration.json"),
        report_sha256=sha(output/"report.json"),package_manifest_sha256=sha(package/"delivery_manifest.json"),
        checkpoint_sha256=manifest["checkpoint_sha256"],raw_identity=identity,
        frames=len(frames),joint_actions=len(pacing),orphan_wire_files=orphan,unsafe_endpoint_indices=unsafe,
        initial_signature_passed=initial_check["passed"],replay=reconstructed,
        platform_processes_started=0,new_physics_steps=0,training_steps=0,
        high_maneuver_validated=False,formal_worker_admission=False)
    if save:
        with (output/"independent_audit.json").open("x",encoding="utf-8",newline="\n") as stream:
            json.dump(result,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write("\n")
    return result


def export_acmi(output,destination):
    """Export actual sampled poses only, outside the preserved raw directory."""
    output,destination=Path(output).resolve(),Path(destination).resolve()
    if destination.is_relative_to(output):raise ValueError("ACMI export must not alter the sealed raw directory")
    receipt=read(output/"independent_audit.json")
    if (receipt.get("audit_pass") is not True or receipt.get("registration_sha256")!=sha(output/"registration.json")
            or receipt.get("report_sha256")!=sha(output/"report.json") or receipt.get("raw_identity")!=raw_identity(output)):
        raise ValueError("unchanged raw flight replay audit must pass before export")
    frames=lines(output/"frames.jsonl");destination.parent.mkdir(parents=True,exist_ok=True)
    with destination.open("x",encoding="utf-8-sig",newline="\n") as stream:
        stream.write("FileType=text/acmi/tacview\nFileVersion=2.2\n0,ReferenceTime=2026-10-26T00:00:00Z\n")
        for i,row in enumerate(frames):
            obs=row["observation"];stream.write("#"+format(obs["red"]["red_0"]["simulation/sim-time-sec"],".9f")+"\n")
            for uid,side,color,name in ((1,"red","Red","PPO"),(2,"blue","Blue","PID")):
                raw=obs[side][side+"_0"]
                pose=[raw["position/long-gc-deg"],raw["position/lat-geod-deg"],raw["position/h-sl-ft"]*.3048,
                    math.degrees(raw["attitude/roll-rad"]),math.degrees(raw["attitude/pitch-rad"]),raw["attitude/psi-deg"]]
                stream.write(str(uid)+",T="+"|".join(format(v,".10f") for v in pose)
                    +(",Type=Air+FixedWing,Name=F16-"+name+",Color="+color if i==0 else "")+"\n")
    return dict(path=str(destination),sha256=sha(destination),actual_sampled_frames=len(frames),interpolated=False,physics_replayed=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root",type=Path,required=True);parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--acmi",type=Path)
    args=parser.parse_args();root=args.package_root.resolve()
    if str(root) not in sys.path:sys.path.insert(0,str(root))
    result=audit(root,args.output)
    if args.acmi:result["export"]=export_acmi(args.output,args.acmi)
    print(json.dumps(result,ensure_ascii=False,allow_nan=False),flush=True)
    if not result["engineering_pass"]:raise SystemExit(2)


if __name__=="__main__":main()
