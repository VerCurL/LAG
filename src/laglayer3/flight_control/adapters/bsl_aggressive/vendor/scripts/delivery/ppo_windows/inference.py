"""Consume real own-aircraft raw/reference JSONL and run the actual frozen PPO.

No socket, simulator, PID or policy training is started by this entry point.
Each input is {raw, reference, logical_time_sec}; first logical time must be0.
"""
import argparse
import json
from pathlib import Path
import sys


def package_root():return Path(__file__).resolve().parents[3]


def configure_inference_runtime():
    """Match registered CPU/thread1 execution without any warmup inference."""
    import torch
    torch.set_num_threads(1)
    return inference_runtime()


def inference_runtime():
    import torch
    return dict(device="cpu",torch_num_threads=torch.get_num_threads(),
        warmup_actor_forward_evaluations=0,extra_RNG_draws=0)


def require_cpu_policy(policy):
    if policy.device.type!="cpu" or any(parameter.device.type!="cpu" for parameter in policy.actor.parameters()):
        raise ValueError("portable inference requires CPU Actor tensors")


def load_manifest(root):
    import hashlib
    import zipfile
    root=Path(root).resolve()
    path=root/"delivery_manifest.json"
    manifest=json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("status") not in ("trained_endpoint_audited","guarded_adapter_validated") or manifest.get("ready_for_platform_screening") is not True:
        raise ValueError("trained audited new-platform PPO endpoint is not installed; no controller fallback")
    if (not isinstance(manifest.get("checkpoint"),str) or not manifest["checkpoint"]
            or not isinstance(manifest.get("checkpoint_sha256"),str) or len(manifest["checkpoint_sha256"])!=64
            or manifest["runtime_files_sha256"].get(manifest["checkpoint"])!=manifest["checkpoint_sha256"]):
        raise ValueError("installed endpoint path/SHA must be bound by the runtime manifest")
    for name,digest in manifest["runtime_files_sha256"].items():
        path=(root/name).resolve()
        if not path.is_relative_to(root.resolve()) or hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
            raise ValueError("portable runtime/config/checkpoint file changed: "+name)
    if manifest.get("PPO_enabled") is not True or manifest.get("PID_fallback") is not False:
        raise ValueError("actual new-platform PPO delivery identity required")
    adapter=manifest.get("control_adapter")
    if manifest["status"]=="trained_endpoint_audited":
        if adapter is not None or manifest.get("original_C83_delivery_gate_pass") is not True:
            raise ValueError("original failed C83 controller cannot be marked ready to fly")
    else:
        from ....scripts.delivery.ppo_windows.builder import _guard_receipts
        if not isinstance(adapter,dict) or manifest.get("original_C83_delivery_gate_pass") is not False:
            raise ValueError("explicit guarded validation is required; original C83 failure must remain false")
        folder=(root/adapter["validation_folder"]).resolve()
        if not folder.is_relative_to(root):raise ValueError("validation evidence must remain inside the package")
        checked=_guard_receipts(folder,manifest["checkpoint_sha256"],root/adapter["guard_source_path"])
        metadata={k:v for k,v in adapter.items() if k not in ("sealed_input_archive","sealed_input_inventory")}
        if metadata!=checked:raise ValueError("manifest adapter differs from closed C84 evidence")
        for name,digest in checked["validation_receipts_sha256"].items():
            bound=(folder/name).relative_to(root).as_posix()
            if manifest["runtime_files_sha256"].get(bound)!=digest:raise ValueError("C84 receipt must be manifest-bound")
        inventory=adapter["sealed_input_inventory"]
        if set(inventory)!=set(checked["input_identity"]):raise ValueError("C84 sealed input inventory differs")
        archive=(root/adapter["sealed_input_archive"]).resolve()
        if not archive.is_relative_to(root) or archive.relative_to(root).as_posix() not in manifest["runtime_files_sha256"]:
            raise ValueError("C84 sealed input archive must be bound inside the package")
        with zipfile.ZipFile(archive) as stream:
            if set(stream.namelist())!={r["member"] for r in inventory.values()} or len(stream.namelist())!=len(inventory):
                raise ValueError("C84 sealed input archive membership differs")
            for name,record in inventory.items():
                digest=checked["input_identity"][name]
                if record["sha256"]!=digest or hashlib.sha256(stream.read(record["member"])).hexdigest()!=digest:
                    raise ValueError("C84 sealed input bytes differ: "+name)
    return manifest


def controller_identity(manifest):
    from ....competition.flight_control_ppo_delivery import CONTROL_SCHEMA
    adapter=manifest.get("control_adapter")
    if adapter is None:return dict(schema=CONTROL_SCHEMA,adapter=None)
    return dict(schema=adapter["schema"],parameters=adapter["parameters"],
        source_sha256=adapter["guard_source_sha256"],weights_trained_with_adapter=False,
        validation_experiment=adapter["validation_experiment"],
        validation_completion_sha256=adapter["validation_receipts_sha256"]["completion.json"])


def controller_from_manifest(root,manifest):
    from ....competition.flight_control_ppo_delivery.config import load_config
    config=load_config(Path(root)/manifest["controller_config"])
    adapter=manifest.get("control_adapter")
    if adapter is None:
        from ....competition.flight_control_ppo_delivery.controller import DeliveryController
        return DeliveryController(config)
    from ....scripts.delivery.ppo_windows.guarded_controller import GuardedController,SCHEMA,adapter_config
    if adapter.get("schema")!=SCHEMA or adapter.get("parameters")!=adapter_config():
        raise ValueError("unknown or altered control adapter; no implicit fallback")
    return GuardedController(config)


def actor_from_package(root):
    manifest=load_manifest(root)
    configure_inference_runtime()
    from ....competition.flight_control_ppo_delivery.actor import ActorDelivery,load_policy
    policy=load_policy(root/manifest["checkpoint"],manifest["checkpoint_sha256"])
    require_cpu_policy(policy)
    controller=controller_from_manifest(root,manifest)
    return ActorDelivery(policy,controller),manifest


def infer_records(actor,records):
    """One inference per supplied actual frame, with real GRU continuity."""
    for index,row in enumerate(records):
        if not isinstance(row,dict) or set(row)!={"raw","reference","logical_time_sec"}:
            raise ValueError("input requires explicit raw/reference/logical_time_sec")
        if index==0:actor.reset(row["raw"],row["reference"],row["logical_time_sec"])
        else:actor.observe(row["raw"],row["reference"],row["logical_time_sec"])
        yield actor.act()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root",type=Path,default=package_root())
    parser.add_argument("--input",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args();root=args.package_root.resolve()
    if str(root) not in sys.path:sys.path.insert(0,str(root))
    actor,_=actor_from_package(root)
    from ....competition.flight_control_ppo_delivery.controller import plain
    with args.input.open(encoding="utf-8") as source,args.output.open("x",encoding="utf-8",newline="\n") as sink:
        rows=(json.loads(line) for line in source if line.strip())
        for result in infer_records(actor,rows):sink.write(json.dumps(plain(result),ensure_ascii=False,allow_nan=False)+"\n")


if __name__=="__main__":main()
