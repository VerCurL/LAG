"""Actual Windows Unity TCP demo: red frozen PPO, blue independent M1 PID.

Import/--help creates no process or socket. An audited new-platform checkpoint
and a fresh output directory are required before an explicitly invoked run.
"""
import argparse
import base64
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import struct
import subprocess
import sys
import time


def write(path,value):
    with Path(path).open("x",encoding="utf-8",newline="\n") as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write("\n")


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def psq(value):return "'"+str(value).replace("'","''")+"'"


def powershell(script):
    encoded=base64.b64encode(("$ErrorActionPreference='Stop'\n[Console]::OutputEncoding=[System.Text.UTF8Encoding]::new()\n"+script).encode("utf-16-le")).decode("ascii")
    result=subprocess.run(["powershell.exe","-NoProfile","-NonInteractive","-EncodedCommand",encoded],
        capture_output=True,text=True,encoding="utf-8",errors="replace",timeout=25)
    if result.returncode:raise RuntimeError("scoped Windows process operation failed: "+result.stderr[-1200:])
    return result.stdout.strip()


def arguments(output,host,port):
    return ["Ip="+host,"Port="+str(port),"PlayMode=1","RedNum=1","BlueNum=1","Red=0","Blue=0","Scenes=4",
            "-screen-fullscreen","0","-screen-width","1280","-screen-height","720","-logFile",str(output/"unity.log")]


def partial_identity(output,executable,args):
    """Never authorize cleanup from a PID alone or a guessed creation time."""
    try:
        pid_text=(output/"launched_pid.txt").read_text(encoding="utf-8-sig").strip()
        seed=json.loads((output/"launch_identity_seed.json").read_text(encoding="utf-8-sig"))
        if (not pid_text.isascii() or not pid_text.isdecimal() or int(pid_text)<=0
                or type(seed.get("pid")) is not int or seed["pid"]!=int(pid_text)
                or not isinstance(seed.get("creation"),str) or not seed["creation"]
                or not isinstance(seed.get("executable"),str) or Path(seed["executable"]).resolve()!=executable):return None
        return dict(seed,arguments=args,identity_verified=False,recovered_from_partial_receipts=True)
    except (OSError,ValueError,TypeError):return None


def launch(executable,output,args):
    script=("$expected="+psq(executable)+"\n"
        +"$p=Start-Process -FilePath $expected -WorkingDirectory "+psq(executable.parent)
        +" -ArgumentList "+psq(subprocess.list2cmdline(args))+" -WindowStyle Normal -PassThru\n"
        +"$launchPid=$p.Id; $creation=$p.StartTime.ToUniversalTime().ToString('o')\n"
        +"$launchPid | Set-Content -LiteralPath "+psq(output/"launched_pid.txt")+" -Encoding ASCII\n"
        +"$seed=[pscustomobject]@{pid=$launchPid;creation=$creation;executable=$expected}\n"
        +"$seed | ConvertTo-Json | Set-Content -LiteralPath "+psq(output/"launch_identity_seed.json")+" -Encoding UTF8\n"
        +"$stable=0;$watch=[System.Diagnostics.Stopwatch]::StartNew()\n"
        +"while ($watch.Elapsed.TotalSeconds -lt 10) {\n"
        +" $current=Get-Process -Id $launchPid -ErrorAction SilentlyContinue\n"
        +" if ($null -eq $current) {throw 'launched process exited'}\n"
        +" if ($current.Path -ieq $expected -and $current.StartTime.ToUniversalTime().ToString('o') -ceq $creation) {$stable++} else {$stable=0}\n"
        +" if ($stable -ge 2) {break};Start-Sleep -Milliseconds 100\n}\n"
        +"if ($stable -lt 2) {throw 'process identity did not become ready'}\n"
        +"[pscustomobject]@{pid=$launchPid;creation=$creation;executable=$expected;identity_verified=$true}|ConvertTo-Json -Compress")
    identity=json.loads(powershell(script));identity["arguments"]=args
    return identity


def stop_owned(identity):
    return powershell("$p=Get-Process -Id "+str(identity["pid"])+" -ErrorAction SilentlyContinue\n"
        +"if ($null -eq $p) {'already_exited';exit 0}\n"
        +"if ($p.Path -ine "+psq(identity["executable"])+" -or $p.StartTime.ToUniversalTime().ToString('o') -cne "+psq(identity["creation"])+") {throw 'PID identity changed; refusing cleanup'}\n"
        +"Stop-Process -Id "+str(identity["pid"])+" -Force -ErrorAction Stop\n"
        +"Wait-Process -Id "+str(identity["pid"])+" -Timeout 5 -ErrorAction SilentlyContinue\n'owned_process_stopped'")


def owned_listener(pid,port):
    data=subprocess.run(["netstat.exe","-ano","-p","TCP"],capture_output=True,text=True,errors="replace",timeout=5).stdout
    for line in data.splitlines():
        p=line.split()
        if len(p)>=5 and p[0]=="TCP" and p[-1]==str(pid) and p[1].rsplit(":",1)[-1]==str(port) and p[2].rsplit(":",1)[-1]=="0":return True
    return False


def exact(connection,count):
    result=bytearray()
    while len(result)<count:
        part=connection.recv(count-len(result))
        if not part:raise EOFError("platform closed before completing a real packet")
        result.extend(part)
    return bytes(result)


def exchange(connection,request,output,index):
    compressed=gzip.compress(json.dumps(request,ensure_ascii=False,allow_nan=False).encode("utf-8"),mtime=0)
    packet=struct.pack("!I",len(compressed))+compressed
    (output/"wire"/("%04d.request.bin"%index)).write_bytes(packet)
    connection.sendall(packet);header=exact(connection,4);size=struct.unpack("!I",header)[0]
    if not 0<size<=16*1024*1024:raise ValueError("invalid bounded platform response length")
    body=exact(connection,size)
    (output/"wire"/("%04d.response.bin"%index)).write_bytes(header+body)
    return json.loads(gzip.decompress(body).decode("utf-8"),parse_constant=lambda v:(_ for _ in ()).throw(ValueError(v)))


def initial_request(initial):
    request={}
    for side in ("red","blue"):
        raw=initial[side][side+"_0"]
        request[side]={side+"_0":{k:v for k,v in raw.items() if k.startswith("ic/")}}
    request["flag"]={"init":{"save":0,"SplitScreen":0}}
    return request


def public_reference(profile,index,heading,height,ground_speed):
    """Limited registered demo tasks; no post-flight reference selection."""
    if profile not in ("hold","heading30","height100") or type(index) is not int or not 0<=index<600:
        raise ValueError("registered120s hold/heading30/height100 demonstration required")
    offset_h=offset_z=0.
    if profile=="heading30":
        offset_h=30. if 50<=index<200 else (-30. if 200<=index<350 else 0.)
    if profile=="height100":
        offset_z=100. if 50<=index<200 else (-100. if 200<=index<350 else 0.)
    return dict(heading_rad=(heading+math.radians(offset_h))%(2*math.pi),height_m=height+offset_z,
        ground_speed_mps=ground_speed,gamma_ref_rad=0.,rates=[0.,0.,0.],preview_gamma_rates_degps=None)


def check_observation(observation,contract):
    """Check every actual endpoint including the final no-more-actions frame."""
    for side in ("red","blue"):
        raw=observation[side][side+"_0"]
        if (isinstance(raw["LifeCurrent"],bool) or isinstance(raw["DeathEvent"],bool)
                or not math.isfinite(raw["LifeCurrent"]) or raw["LifeCurrent"]<=0 or raw["DeathEvent"]!=99):
            raise ValueError("actual life/death event: "+side)
        values=dict(height_m=raw["position/h-sl-ft"]*.3048,eas_mps=raw["velocities/ve-fps"]*.3048,
            alpha_deg=raw["aero/alpha-deg"],beta_deg=raw["aero/beta-deg"],roll_deg=math.degrees(raw["attitude/roll-rad"]),pitch_deg=math.degrees(raw["attitude/pitch-rad"]))
        if any(not math.isfinite(values[k]) or not a<=values[k]<=b for k,(a,b) in contract["observed_domain"].items()):
            raise ValueError("actual observable-domain exit: "+side)


def run(package,executable,output,profile="hold",port=18383):
    if os.name!="nt":raise OSError("actual Windows platform required")
    package,executable,output=Path(package).resolve(),Path(executable).resolve(),Path(output).resolve()
    if not executable.is_file() or executable.name.lower()!="zk.exe":raise ValueError("actual authorized ZK.exe path required")
    if output.exists():raise FileExistsError("fresh registered demo output required; no same-identity retry")
    from ....scripts.delivery.ppo_windows.inference import actor_from_package,inference_runtime,controller_identity
    from ....competition.flight_control_ppo_delivery.controller import plain
    from ....competition.flight_control_eas import runtime,evaluation
    from ....competition.flight_control_eas.platform_clock_v1 import Binary32ControlClock,controller_state
    from ....competition.flight_control_eas.platform_reset_v1 import validate_initial,DEFAULT_TOLERANCES
    from ....competition.flight_control_eas_development.pid_speed_proportional import make_controller
    actor,manifest=actor_from_package(package)  # Must succeed before any process/socket.
    original=json.loads((package/"assets/pid_model.json").read_text(encoding="utf-8"))
    initial=json.loads((package/"assets/platform_initial.json").read_text(encoding="utf-8"))
    captured_init=json.loads((package/"assets/platform_init_request.json").read_text(encoding="utf-8"))
    contract=json.loads((package/"assets/platform_contract.json").read_text(encoding="utf-8"))
    if profile not in ("hold","heading30","height100") or type(port) is not int or not 1024<=port<=65535:
        raise ValueError("supported preregistered profile and user-selected TCP port required")
    output.mkdir(parents=True);(output/"wire").mkdir()
    write(output/"registration.json",dict(schema="portable-platform-PPO-demo-v1",profile=profile,port=port,
        package_manifest_sha256=sha(package/"delivery_manifest.json"),checkpoint_sha256=manifest["checkpoint_sha256"],
        executable=str(executable),executable_sha256=sha(executable),duration_sec=120.,
        expected_frames=601,expected_joint_actions=600,red_actual_Actor=True,blue_independent_PID=True,
        inference_runtime=inference_runtime(),
        controller_identity=controller_identity(manifest),
        PID_fallback=False,high_maneuver_claim=False,reference_kind="groundspeed,heading,height; explicit zero target rates and no plan preview"))
    streams={n:(output/n).open("x",encoding="utf-8",newline="\n") for n in ("frames.jsonl","decisions.jsonl","clock_receipts.jsonl","pacing.jsonl")}
    identity=connection=cleanup=error=None;frames=actions=0;start=time.monotonic();args=arguments(output,"127.0.0.1",port);launch_attempted=False
    def line(name,value):streams[name].write(json.dumps(plain(value),ensure_ascii=False,allow_nan=False)+"\n");streams[name].flush()
    try:
        with socket.socket(socket.AF_INET,socket.SOCK_STREAM) as check:
            if hasattr(socket,"SO_EXCLUSIVEADDRUSE"):check.setsockopt(socket.SOL_SOCKET,socket.SO_EXCLUSIVEADDRUSE,1)
            check.bind(("127.0.0.1",port))
        launch_attempted=True;identity=launch(executable,output,args);write(output/"launched_process.json",identity)
        while time.monotonic()-start<50.:
            if time.monotonic()-start>=20. and owned_listener(identity["pid"],port):break
            time.sleep(.5)
        else:raise TimeoutError("owned platform listener did not become ready")
        connection=socket.socket(socket.AF_INET,socket.SOCK_STREAM);connection.settimeout(10.);connection.connect(("127.0.0.1",port))
        obs=exchange(connection,captured_init,output,0);frames=1;line("frames.jsonl",dict(index=0,observation=obs))
        check_observation(obs,contract)
        checked=validate_initial(obs,initial,DEFAULT_TOLERANCES);write(output/"initial_signature.json",checked)
        if not checked["passed"]:raise ValueError("observable platform initialization differs; no action permitted")
        clock=Binary32ControlClock();receipt=clock.observe(obs,0);line("clock_receipts.jsonl",receipt)
        blue=make_controller("pid",original);blue.reset()
        bases={side:dict(heading=math.radians(obs[side][side+"_0"]["attitude/psi-deg"]),height=obs[side][side+"_0"]["position/h-sl-ft"]*.3048,
            ground_speed=160.02) for side in ("red","blue")}
        pacing_origin=time.monotonic()
        for index in range(600):
            if time.monotonic()-start>240.:raise TimeoutError("finite platform demo wall budget exhausted")
            refs={s:public_reference(profile,index,**bases[s]) for s in bases}
            if index==0:actor.reset(obs["red"]["red_0"],refs["red"],0.)
            else:actor.observe(obs["red"]["red_0"],refs["red"],index*.2)
            red=actor.act()
            state=runtime.parse_aircraft(obs["blue"]["blue_0"]);logical=controller_state(state,receipt,"blue")
            ground=math.sqrt(state.v_north_mps**2+state.v_east_mps**2+state.v_down_mps**2)
            eas_target=refs["blue"]["ground_speed_mps"]*state.eas_mps/ground
            blue_ref=runtime.make_reference(dict(heading_deg=math.degrees(refs["blue"]["heading_rad"]),height_m=refs["blue"]["height_m"],eas_mps=eas_target))
            blue_out,inner=blue.step(logical,blue_ref,.2)
            red_request=dict(zip(("fcs/aileron-cmd-norm","fcs/elevator-cmd-norm","fcs/rudder-cmd-norm","fcs/throttle-cmd-norm"),map(float,red["command"])),mode=0,**{"fcs/weapon-launch":0,"switch-missile":0,"change-target":9})
            request=dict(red=dict(red_0=red_request),blue=dict(blue_0=evaluation.complete_action(blue_out)))
            line("decisions.jsonl",dict(index=index,reference=refs,red_actual_PPO=red,blue_PID_output=plain(blue_out.__dict__),blue_inner_reference=plain(inner.__dict__),request=request))
            deadline=pacing_origin+(index+1)*.2;remaining=deadline-time.monotonic()
            if remaining>0:time.sleep(remaining)
            sent=time.monotonic()-pacing_origin
            obs=exchange(connection,request,output,index+1);actions+=1;frames+=1
            line("pacing.jsonl",dict(index=index+1,scheduled_offset_sec=(index+1)*.2,send_offset_sec=sent,receive_offset_sec=time.monotonic()-pacing_origin))
            line("frames.jsonl",dict(index=index+1,observation=obs));check_observation(obs,contract)
            receipt=clock.observe(obs,index+1);line("clock_receipts.jsonl",receipt)
    except Exception as exc:error=dict(type=type(exc).__name__,message=str(exc))
    finally:
        for stream in streams.values():stream.close()
        if connection is not None:connection.close()
        if identity is None:identity=partial_identity(output,executable,args)
        if identity is not None:
            try:cleanup=stop_owned(identity)
            except Exception as exc:cleanup=dict(manual_recovery_required=True,message=str(exc))
        elif launch_attempted:cleanup=dict(manual_recovery_required=True,reason="no complete creation identity; refusing PID-only cleanup")
    from ....scripts.delivery.ppo_windows.inference import load_manifest
    try:load_manifest(package);runtime_unchanged=True
    except Exception as exc:
        runtime_unchanged=False
        if error is None:error=dict(type=type(exc).__name__,message=str(exc))
    executable_unchanged=sha(executable)==json.loads((output/"registration.json").read_text(encoding="utf-8"))["executable_sha256"]
    complete=error is None and frames==601 and actions==600
    report=dict(schema="portable-platform-PPO-demo-v1",frames=frames,joint_actions=actions,
        actual_actor_forward_evaluations=actor.actor_forward_evaluations,complete=complete,
        engineering_pass=bool(complete and runtime_unchanged and executable_unchanged and cleanup in ("owned_process_stopped","already_exited")),
        runtime_unchanged=runtime_unchanged,executable_unchanged=executable_unchanged,
        error=error,cleanup=cleanup,launched_process=identity,PID_fallback=False,high_maneuver_validated=False,
        independent_audit_performed=False,not_a_formal_competition_admission=True)
    report["registration_sha256"]=sha(output/"registration.json")
    report["raw_identity"]={p.relative_to(output).as_posix():sha(p) for p in output.rglob("*") if p.is_file()
        and p.name not in {"registration.json","report.json","independent_audit.json","independent_audit_failure.json","completion.json"}}
    write(output/"report.json",report);return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root",type=Path,required=True);parser.add_argument("--platform-exe",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True);parser.add_argument("--profile",choices=("hold","heading30","height100"),default="hold")
    parser.add_argument("--port",type=int,default=18383)
    args=parser.parse_args();root=args.package_root.resolve()
    if str(root) not in sys.path:sys.path.insert(0,str(root))
    report=run(root,args.platform_exe,args.output,args.profile,args.port)
    print(json.dumps(report,ensure_ascii=False,allow_nan=False),flush=True)
    if not report["engineering_pass"]:raise SystemExit(2)


if __name__=="__main__":main()
