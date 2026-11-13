"""Create a standalone source package; no training, flight or fake weights."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[3]


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_paths():
    paths={ROOT/"config.py",ROOT/"competition/__init__.py"}
    for prefix in ("envs","algorithms","competition/flight_control_ppo_delivery","competition/flight_control_eas"):
        paths.update((ROOT/prefix).rglob("*.py"))
    paths.update((ROOT/"competition/flight_control_ppo_delivery").glob("*.json"))
    paths.add(ROOT/"competition/flight_control_eas/pid_config.json")
    for name in ("competition/flight_control_eas_development/__init__.py",
        "competition/flight_control_eas_development/pid_speed_proportional.py",
        "competition/flight_control_agent/__init__.py","competition/flight_control_agent/bridge.py",
        "scripts/experiments/c46_compact_policy.py","scripts/experiments/c46_compact_observation.py",
        "scripts/experiments/c45_policy_observation.py"):
        paths.add(ROOT/name)
    paths.update(Path(__file__).parent.glob("*.py"))
    return sorted(p for p in paths if p.is_file() and "/data/" not in p.as_posix())


def _write(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")


def _process_receipts(folder,audit,done,*,audit_uses_filenames=True):
    """C83 audit binds filenames; completion binds named SHA fields."""
    for key,name in (("launch_started_sha256","launch_started.json"),("process_exit_sha256","process_exit.json"),("process_log_sha256","process.log")):
        digest=sha(folder/name)
        audit_key=name if audit_uses_filenames else key
        if done.get(key)!=digest or audit["process"].get(audit_key)!=digest:
            raise ValueError("audited worker receipt changed")
    start,end=[json.loads((folder/n).read_text(encoding="utf-8")) for n in ("launch_started.json","process_exit.json")]
    if (end.get("exit_code")!=0 or end.get("pid")!=start.get("pid")
            or end.get("launch_started_sha256")!=sha(folder/"launch_started.json")
            or end.get("process_log_sha256")!=sha(folder/"process.log")
            or start.get("registration_sha256")!=sha(folder/"registration.json")):
        raise ValueError("normal supervised formal training required")


def _guard_receipts(folder,checkpoint_sha,guard_source):
    """Verify C84's explicit schema; never accepts a C83 failure as a pass."""
    from ....scripts.delivery.ppo_windows.guarded_controller import adapter_config,SCHEMA as ADAPTER_SCHEMA
    names=("registration.json","registered_sources.zip","report.json","independent_audit.json","completion.json",
        "launch_started.json","process_exit.json","process.log")
    reg,report,audit,done=[json.loads((folder/name).read_text(encoding="utf-8")) for name in
        ("registration.json","report.json","independent_audit.json","completion.json")]
    if (any(r.get("experiment")!="C84-V0" for r in (reg,report,audit,done))
            or reg.get("schema")!="c84-guarded-platform-validation-v1"
            or report.get("schema")!="c84-guarded-platform-validation-v1"
            or audit.get("schema")!="c84-independent-guarded-platform-validation-audit-v1"
            or audit.get("audit_pass") is not True or done.get("audit_pass") is not True
            or done.get("status")!="completed_independent_audit_passed"
            or any(r.get("registration_sha256")!=sha(folder/"registration.json") for r in (report,audit,done))
            or any(r.get("report_sha256")!=sha(folder/"report.json") for r in (audit,done))
            or done.get("independent_audit_sha256")!=sha(folder/"independent_audit.json")
            or any((folder/name).exists() for name in ("failure.json","independent_audit_failure.json"))):
        raise ValueError("closed C84 independent validation receipt graph required")
    comparison=report.get("comparison",{})
    if (comparison.get("delivery_gate_pass") is not True or done.get("delivery_gate_pass") is not True
            or audit.get("delivery_gate_pass") is not True
            or comparison.get("weights_trained_with_adapter") is not False
            or comparison.get("original_C83_delivery_gate_pass") is not False
            or comparison.get("weight_tensors_modified") is not False
            or not comparison.get("checks") or not all(v is True for v in comparison["checks"].values())):
        raise ValueError("C84 guarded delivery gates must pass without rewriting original C83 failure")
    contract=reg["plan"]["contract"]
    if (contract.get("adapter")!=adapter_config() or contract.get("checkpoint_sha256")!=checkpoint_sha
            or contract.get("weights_trained_with_adapter") is not False
            or contract.get("original_C83_delivery_gate_pass") is not False):
        raise ValueError("C84 adapter parameters or frozen256 weight identity differ")
    raw=report.get("raw_artifact_sha256")
    if not raw or raw!=audit.get("raw_artifact_sha256") or raw!=done.get("raw_artifact_sha256"):
        raise ValueError("C84 audited raw inventories differ")
    if any(report.get("budget",{}).get(k)!=0 for k in ("training_decisions","training_iterations","optimizer_steps","critic_forward_evaluations")):
        raise ValueError("C84 must validate frozen weights without training or critic forward")
    _process_receipts(folder,audit,done,audit_uses_filenames=False)
    archive=folder/"registered_sources.zip"
    if reg.get("source_snapshot_sha256")!=sha(archive):raise ValueError("C84 registered source archive changed")
    with zipfile.ZipFile(archive) as stream:
        if set(stream.namelist())!=set(reg["source_identity"]):raise ValueError("C84 source archive membership differs")
        for name,digest in reg["source_identity"].items():
            if hashlib.sha256(stream.read(name)).hexdigest()!=digest:raise ValueError("C84 source archive bytes differ")
    guard_name="scripts/delivery/ppo_windows/guarded_controller.py"
    if reg["source_identity"].get(guard_name)!=sha(guard_source):raise ValueError("guard source differs from independently validated controller")
    return dict(schema=ADAPTER_SCHEMA,parameters=adapter_config(),weights_trained_with_adapter=False,
        guard_source_path=guard_name,guard_source_sha256=sha(guard_source),validation_experiment="C84-V0",
        checkpoint_sha256=checkpoint_sha,original_C83_delivery_gate_pass=False,
        validation_receipts_sha256={name:sha(folder/name) for name in names},
        validation_folder="assets/c84_validation",input_identity=reg["input_identity"])


def _guard_validation_graph(completion,checkpoint_sha):
    if completion.name!="completion.json":raise ValueError("C84 completion.json is required")
    folder=completion.parent
    descriptor=_guard_receipts(folder,checkpoint_sha,ROOT/"scripts/delivery/ppo_windows/guarded_controller.py")
    inputs={}
    for name,digest in descriptor["input_identity"].items():
        path=Path(name) if Path(name).is_absolute() else ROOT/name
        if sha(path)!=digest:raise ValueError("registered C84 input changed: "+name)
        inputs[name]=path
    return descriptor,folder,inputs


def _receipt_graph(completion,checkpoint,checkpoint_sha):
    folder=completion.parent
    names=("registration.json","report.json","independent_audit.json","completion.json")
    values={n:json.loads((folder/n).read_text(encoding="utf-8")) for n in names}
    reg,report,audit,done=[values[n] for n in names]
    if (completion.name!="completion.json" or reg.get("experiment")!="C83-R0" or reg.get("mode")!="R0"
            or report.get("experiment")!="C83-R0" or report.get("mode")!="R0"
            or report.get("actor_best_created") is not True or report.get("smoke_resume_allowed") is not False
            or done.get("experiment")!="C83-R0" or done.get("status")!="completed_independent_audit_passed"
            or done.get("audit_pass") is not True or audit.get("audit_pass") is not True
            or done.get("registration_sha256")!=sha(folder/"registration.json")
            or report.get("registration_sha256")!=sha(folder/"registration.json")
            or audit.get("registration_sha256")!=sha(folder/"registration.json")
            or done.get("report_sha256")!=sha(folder/"report.json")
            or audit.get("report_sha256")!=sha(folder/"report.json")
            or done.get("independent_audit_sha256")!=sha(folder/"independent_audit.json")):
        raise ValueError("completed independent training/endpoint receipt graph required")
    selected=folder/"training/best_checkpoint.json"
    endpoint=json.loads(selected.read_text(encoding="utf-8"))
    source=folder/"training/update_255/after.pt"
    if (checkpoint.resolve()!=(folder/"training/actor_best.pt").resolve()
            or sha(checkpoint)!=checkpoint_sha or sha(source)!=checkpoint_sha
            or report["training"].get("actor_best_sha256")!=checkpoint_sha
            or report["training"].get("best_checkpoint")!=endpoint
            or endpoint.get("experiment")!="C83-R0" or endpoint.get("chosen_completed_updates")!=256
            or endpoint.get("actor_best_sha256")!=checkpoint_sha or endpoint.get("source_checkpoint_sha256")!=checkpoint_sha
            or endpoint.get("source_checkpoint")!="update_255/after.pt"
            or endpoint.get("selection_rule")!="predeclared_update255_budget_endpoint_no_ranking"
            or endpoint.get("eligible_completed_updates")!=[256]
            or endpoint.get("training_completed") is not True or endpoint.get("resume_authorized") is not False
            or endpoint.get("smoke_initialization_loaded") is not False
            or endpoint.get("development_ranking_used") is not False
            or endpoint.get("final_cases_used_for_selection") is not False
            or endpoint.get("endpoint_failure_fallback") is not False):
        raise ValueError("only the audited fresh256 fixed C83-R0 endpoint may be packaged")
    artifacts=report["raw_artifact_sha256"]
    if artifacts!=audit.get("raw_artifact_sha256") or artifacts!=done.get("raw_artifact_sha256"):
        raise ValueError("endpoint raw artifact receipts disagree")
    for name in ("training/actor_best.pt","training/update_255/after.pt","training/best_checkpoint.json"):
        if artifacts.get(name)!=sha(folder/name):raise ValueError("audited endpoint artifact changed: "+name)
    _process_receipts(folder,audit,done)
    archive=folder/"registered_sources.zip"
    if reg.get("source_snapshot_sha256")!=sha(archive):raise ValueError("audited source archive changed")
    with zipfile.ZipFile(archive) as stream:
        if set(stream.namelist())!=set(reg["source_identity"]):raise ValueError("training source archive membership differs")
        for name,digest in reg["source_identity"].items():
            if hashlib.sha256(stream.read(name)).hexdigest()!=digest:raise ValueError("training source archive bytes differ")
    # Shared runtime modules must be the exact sources used by the audited run;
    # separate portable tools can be added without rewriting that experiment.
    for path in source_paths():
        name=path.relative_to(ROOT).as_posix()
        if name in reg["source_identity"] and sha(path)!=reg["source_identity"][name]:
            raise ValueError("runtime source differs from audited training: "+name)
    return [(folder/n) for n in names]+[selected,archive]+[folder/n for n in ("launch_started.json","process_exit.json","process.log")]


def prepare(destination,checkpoint=None,expected_sha256=None,completion=None,adapter_validation_receipt=None):
    destination=Path(destination).resolve()
    if destination.exists():raise FileExistsError("new package directory required")
    receipts=[];adapter=None;adapter_folder=None;adapter_inputs={};base_gate=False
    if checkpoint is not None:
        from ....competition.flight_control_ppo_delivery.actor import load_policy
        if completion is None:raise ValueError("audited endpoint completion receipt is required before installing weights")
        checkpoint=Path(checkpoint).resolve();load_policy(checkpoint,expected_sha256)
        receipts=_receipt_graph(Path(completion).resolve(),checkpoint,expected_sha256)
        base_report=json.loads((Path(completion).resolve().parent/"report.json").read_text(encoding="utf-8"))
        base_gate=base_report["evaluation"]["comparison"]["delivery_gate_pass"] is True
        if adapter_validation_receipt is not None:
            adapter,adapter_folder,adapter_inputs=_guard_validation_graph(Path(adapter_validation_receipt).resolve(),expected_sha256)
    elif expected_sha256 is not None or completion is not None or adapter_validation_receipt is not None:
        raise ValueError("checkpoint, expected SHA and completed audit receipts must be supplied together")
    destination.mkdir(parents=True,exist_ok=False)
    files={}
    for path in source_paths():
        name=path.relative_to(ROOT);target=destination/name
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,target)
        files[name.as_posix()]=sha(target)
    # Namespace packages are explicit and contain no executable launch code.
    for name in ("scripts/__init__.py","scripts/experiments/__init__.py","scripts/delivery/__init__.py"):
        path=destination/name
        if not path.exists():path.parent.mkdir(parents=True,exist_ok=True);path.write_text("",encoding="utf-8");files[name]=sha(path)
    assets=destination/"assets";assets.mkdir()
    parent=ROOT/"scripts/results/diagnostics/2026-10-23-C62-V3-pid-platform-interactive"
    reg=json.loads((parent/"registration.json").read_text(encoding="utf-8"))
    shutil.copyfile(reg["model_path"],assets/"pid_model.json")
    with Path(reg["initial_reference_path"]).open(encoding="utf-8") as stream:initial=json.loads(next(stream))["observation"]
    _write(assets/"platform_initial.json",initial)
    _write(assets/"platform_contract.json",reg["protocol"])
    captured=(parent/"wire/000.request.bin").read_bytes()
    _write(assets/"platform_init_request.json",json.loads(gzip.decompress(captured[4:]).decode("utf-8")))
    for p in assets.iterdir():files[p.relative_to(destination).as_posix()]=sha(p)
    if checkpoint is not None:
        shutil.copyfile(checkpoint,assets/"actor_best.pt");files["assets/actor_best.pt"]=sha(assets/"actor_best.pt")
        for p in receipts:
            target=assets/("training_"+p.name);shutil.copyfile(p,target)
            files[target.relative_to(destination).as_posix()]=sha(target)
    if adapter is not None:
        validation=destination/adapter["validation_folder"];validation.mkdir()
        for name in adapter["validation_receipts_sha256"]:
            target=validation/name;shutil.copyfile(adapter_folder/name,target)
            files[target.relative_to(destination).as_posix()]=sha(target)
        inventory={}
        with zipfile.ZipFile(validation/"registered_inputs.zip","x",zipfile.ZIP_DEFLATED) as stream:
            for i,(name,path) in enumerate(sorted(adapter_inputs.items())):
                member="%04d/"%i+path.name
                digest=sha(path)
                if digest!=adapter["input_identity"][name]:raise ValueError("C84 input changed while packaging")
                stream.write(path,member);inventory[name]=dict(member=member,sha256=digest)
        target=validation/"registered_inputs.zip"
        files[target.relative_to(destination).as_posix()]=sha(target)
        adapter["sealed_input_inventory"]=inventory
        adapter["sealed_input_archive"]="assets/c84_validation/registered_inputs.zip"
    shutil.copyfile(Path(__file__).with_name("requirements-windows-observed.txt"),destination/"requirements-windows-observed.txt")
    for name in ("README.md","DELIVERY_VALIDATION.md","PLATFORM_FLIGHT_PLAN.md"):
        shutil.copyfile(Path(__file__).with_name(name),destination/name)
        files[name]=sha(destination/name)
    files["requirements-windows-observed.txt"]=sha(destination/"requirements-windows-observed.txt")
    manifest=dict(schema="windows-platform-residual-PPO-delivery-v1",
        status=("prepared_without_weights" if checkpoint is None else "guarded_adapter_validated" if adapter is not None
            else "trained_endpoint_audited" if base_gate else "trained_endpoint_audited_not_ready"),
        PPO_enabled=True,PID_fallback=False,legacy_weights_compatible=False,
        checkpoint=None if checkpoint is None else "assets/actor_best.pt",checkpoint_sha256=expected_sha256,
        controller_config="competition/flight_control_ppo_delivery/default_config.json",
        control_adapter=adapter,original_C83_delivery_gate_pass=base_gate,
        ready_for_platform_screening=bool(checkpoint is not None and (base_gate or adapter is not None)),
        platform_performance_certified=False,
        runtime_files_sha256=dict(sorted(files.items())),
        actual_platform_flight_passed=False,actual_platform_flight_not_performed_by_builder=True,
        speed_reference_kind="ground speed magnitude m/s, not EAS",training_or_flight_started=False)
    _write(destination/"delivery_manifest.json",manifest)
    return manifest


def archive_package(package,destination):
    """Write a new ZIP from a fully validated trained package, without a flight."""
    from ....scripts.delivery.ppo_windows.inference import load_manifest
    package,destination=Path(package).resolve(),Path(destination).resolve()
    if destination.is_relative_to(package):raise ValueError("archive must be outside the package directory")
    if destination.exists():raise FileExistsError("new ZIP path required")
    manifest=load_manifest(package)
    expected=set(manifest["runtime_files_sha256"])|{"delivery_manifest.json"}
    actual={p.relative_to(package).as_posix() for p in package.rglob("*") if p.is_file()}
    if actual!=expected:raise ValueError("unbound extra or missing files in delivery package")
    destination.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(destination,"x",zipfile.ZIP_DEFLATED) as stream:
        for name in sorted(expected):stream.write(package/name,name)
    with zipfile.ZipFile(destination) as stream:
        if set(stream.namelist())!=expected or stream.testzip() is not None:raise ValueError("delivery ZIP verification failed")
        for name in expected:
            if hashlib.sha256(stream.read(name)).hexdigest()!=sha(package/name):raise ValueError("delivery ZIP byte mismatch")
    load_manifest(package)
    return dict(path=str(destination),sha256=sha(destination),files=len(expected),
        checkpoint_sha256=manifest["checkpoint_sha256"],actual_platform_flight_performed=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--checkpoint",type=Path)
    parser.add_argument("--expected-sha256")
    parser.add_argument("--completion",type=Path)
    parser.add_argument("--adapter-validation-receipt",type=Path,help="closed, passing C84-V0 completion.json for explicit guarded control")
    parser.add_argument("--zip",type=Path,help="optional new ZIP path, only for an audited trained endpoint")
    args=parser.parse_args()
    if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
    result=prepare(args.output,args.checkpoint,args.expected_sha256,args.completion,args.adapter_validation_receipt)
    if args.zip:print(json.dumps(archive_package(args.output,args.zip),ensure_ascii=False),flush=True)
    print(result["status"],len(result["runtime_files_sha256"]),flush=True)


if __name__=="__main__":main()
