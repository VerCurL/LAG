#!/usr/bin/env python
import argparse
import csv
import json
import logging
import random
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

logging.basicConfig(level=logging.INFO, format="%(message)s")

try:
    import gymnasium as gym
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
except ModuleNotFoundError as exc:
    logging.info(f"Error: missing dependency: {exc}")
    logging.info("Please activate the same Python environment used by this project, then run this script again.")
    sys.exit(1)

from algorithms.utils.AeroTAF_ATTN import AeroTAFATTNBase
from scripts.AeroTAF.collector.path_utils import normalize_path, resolve_project_path


THREAT_HEAD_PATTERN = "threat_output_module"
ATTACK_HEAD_PATTERN = "attack_output_module"


class AeroTAFATTNArgs:
    def __init__(self, args):
        self.num_agents = args.num_agents
        self.activation_id = args.activation_id
        self.use_feature_normalization = args.use_feature_normalization
        self.num_heads = args.num_heads
        self.time_head_num = args.time_head_num
        self.KQ_hidden_size = args.KQ_hidden_size
        self.V_hidden_size = args.V_hidden_size
        self.attn_output_hidden_size = args.attn_output_hidden_size
        self.field_output_hidden_size = args.field_output_hidden_size


class PPOAeroTAFATTN(nn.Module):
    def __init__(self, args, obs_space, act_space, device=torch.device("cpu")):
        super().__init__()
        self.num_agents = args.num_agents
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.AeroTAF = AeroTAFATTNBase(
            obs_space=obs_space,
            act_space=act_space,
            agent_num=args.num_agents,
            head_num=args.num_heads,
            time_head_num=args.time_head_num,
            KQ_hidden_size=args.KQ_hidden_size,
            V_hidden_size=args.V_hidden_size,
            attn_output_hidden_size=args.attn_output_hidden_size,
            field_output_hidden_size=args.field_output_hidden_size,
            activation_id=args.activation_id,
            use_feature_normalization=args.use_feature_normalization,
        )
        self.to(device)

    def forward(self, obs, actions, seq_len=None, time_offset=0):
        obs = torch.as_tensor(obs, **self.tpdv)
        actions = torch.as_tensor(actions, **self.tpdv)
        return self.AeroTAF(obs, actions, seq_len=seq_len, time_offset=time_offset)


class AeroTAFWindowStore(Dataset):
    def __init__(self, npz_path, use_temporal_loss=True):
        with np.load(npz_path, allow_pickle=True) as data:
            required = ["obs", "actions", "threat_targets", "attack_targets"]
            if use_temporal_loss:
                required.append("temporal_targets")
            missing = [key for key in required if key not in data.files]
            if missing:
                raise KeyError(f"{npz_path} missing keys: {missing}")

            self.obs = data["obs"].astype(np.float32, copy=False)
            self.actions = data["actions"].astype(np.float32, copy=False)
            self.threat_targets = data["threat_targets"].astype(np.float32, copy=False)
            self.attack_targets = data["attack_targets"].astype(np.float32, copy=False)
            self.temporal_targets = (
                data["temporal_targets"].astype(np.float32, copy=False)
                if "temporal_targets" in data.files
                else None
            )
            self.window_lengths = (
                data["window_lengths"].astype(np.int32, copy=False)
                if "window_lengths" in data.files
                else np.full((self.obs.shape[0],), self.obs.shape[1], dtype=np.int32)
            )

        if self.obs.ndim != 4:
            raise ValueError(f"{npz_path}: obs must be [windows, time, agents, obs_dim], got {self.obs.shape}")
        if self.actions.ndim != 4:
            raise ValueError(f"{npz_path}: actions must be [windows, time, agents, act_dim], got {self.actions.shape}")
        if self.threat_targets.ndim != 3:
            raise ValueError(f"{npz_path}: threat_targets must be [windows, time, 1], got {self.threat_targets.shape}")
        if self.attack_targets.ndim != 3:
            raise ValueError(f"{npz_path}: attack_targets must be [windows, time, 1], got {self.attack_targets.shape}")

        expected_prefix = self.obs.shape[:2]
        for name, value in (
            ("actions", self.actions),
            ("threat_targets", self.threat_targets),
            ("attack_targets", self.attack_targets),
        ):
            if value.shape[:2] != expected_prefix:
                raise ValueError(f"{npz_path}: {name} prefix {value.shape[:2]} != obs prefix {expected_prefix}")
        if use_temporal_loss and self.temporal_targets is None:
            raise KeyError(f"{npz_path}: temporal_targets is required when temporal loss is enabled.")
        if self.temporal_targets is not None and self.temporal_targets.shape[:3] != self.obs.shape[:3]:
            raise ValueError(f"{npz_path}: temporal_targets prefix {self.temporal_targets.shape[:3]} != obs prefix {self.obs.shape[:3]}")

    def __len__(self):
        return int(self.obs.shape[0])

    def __getitem__(self, index):
        temporal_targets = self.temporal_targets[index] if self.temporal_targets is not None else np.zeros(1, dtype=np.float32)
        return (
            self.obs[index],
            self.actions[index],
            self.threat_targets[index],
            self.attack_targets[index],
            temporal_targets,
        )

    @property
    def window_length(self):
        return int(self.obs.shape[1])

    @property
    def step_count(self):
        return int(self.obs.shape[0] * self.obs.shape[1])


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_device(args):
    if args.cuda and torch.cuda.is_available():
        if args.cuda_device_id >= 0:
            return torch.device(f"cuda:{args.cuda_device_id}")
        return torch.device("cuda:0")
    return torch.device("cpu")


def build_spaces(dataset):
    obs_dim = dataset.obs.shape[-1]
    act_dim = dataset.actions.shape[-1]
    obs_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
    act_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(act_dim,), dtype=np.float32)
    return obs_space, act_space


def is_threat_head_param_name(name):
    return THREAT_HEAD_PATTERN in name


def is_attack_head_param_name(name):
    return ATTACK_HEAD_PATTERN in name


def split_parameter_groups(model):
    backbone_params = []
    threat_head_params = []
    attack_head_params = []
    for name, param in model.named_parameters():
        if is_threat_head_param_name(name):
            threat_head_params.append(param)
        elif is_attack_head_param_name(name):
            attack_head_params.append(param)
        else:
            backbone_params.append(param)
    return backbone_params, threat_head_params, attack_head_params


def zero_param_grads(params):
    for param in params:
        param.grad = None


def assign_grads(params, grads):
    for param, grad in zip(params, grads):
        param.grad = None if grad is None else grad.detach()


def clip_param_grads(params, max_grad_norm):
    if max_grad_norm is not None and max_grad_norm > 0 and params:
        torch.nn.utils.clip_grad_norm_(params, max_grad_norm)


def build_optimizers(model, args):
    backbone_params, threat_head_params, attack_head_params = split_parameter_groups(model)
    if args.train_mode == "split":
        optimizers = {
            "backbone": torch.optim.Adam(backbone_params, lr=args.lr, weight_decay=args.weight_decay),
            "threat_head": torch.optim.Adam(threat_head_params, lr=args.lr, weight_decay=args.weight_decay),
            "attack_head": torch.optim.Adam(attack_head_params, lr=args.lr, weight_decay=args.weight_decay),
        }
        return None, optimizers, backbone_params, threat_head_params, attack_head_params

    if args.train_mode == "heads_only":
        params = threat_head_params + attack_head_params
    else:
        params = list(model.parameters())
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    return optimizer, {}, backbone_params, threat_head_params, attack_head_params


class WindowPartitionBatchSampler:
    def __init__(self, dataset_size, num_batches, seed=1, epoch=0, shuffle=True):
        self.dataset_size = int(dataset_size)
        self.num_batches = max(1, min(int(num_batches), self.dataset_size))
        self.seed = int(seed)
        self.epoch = int(epoch)
        self.shuffle = bool(shuffle)

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        indices = list(range(self.dataset_size))
        if self.shuffle:
            random.Random(self.seed + self.epoch * 9973).shuffle(indices)
        for part in np.array_split(np.asarray(indices, dtype=np.int64), self.num_batches):
            if len(part) > 0:
                yield part.tolist()


def build_train_loader(dataset, args, device, epoch):
    return DataLoader(
        dataset,
        batch_sampler=WindowPartitionBatchSampler(
            dataset_size=len(dataset),
            num_batches=args.effective_mini_epoch,
            seed=args.seed,
            epoch=epoch,
            shuffle=True,
        ),
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )


def build_eval_loader(dataset, args, device):
    return DataLoader(
        dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )


def prepare_batch(batch, device):
    obs, actions, threat_targets, attack_targets, temporal_targets = batch
    obs = torch.as_tensor(obs, dtype=torch.float32, device=device)
    actions = torch.as_tensor(actions, dtype=torch.float32, device=device)
    threat_targets = torch.as_tensor(threat_targets, dtype=torch.float32, device=device)
    attack_targets = torch.as_tensor(attack_targets, dtype=torch.float32, device=device)
    temporal_targets = torch.as_tensor(temporal_targets, dtype=torch.float32, device=device)

    if obs.ndim != 4:
        raise ValueError(f"Expected obs [B,T,N,D], got {tuple(obs.shape)}")
    batch_size, seq_len, num_agents = obs.shape[:3]

    obs = obs.reshape(batch_size * seq_len * num_agents, -1)
    actions = actions.reshape(batch_size * seq_len * num_agents, -1)
    threat_targets = threat_targets.reshape(batch_size * seq_len, -1)
    attack_targets = attack_targets.reshape(batch_size * seq_len, -1)
    if temporal_targets.ndim >= 4:
        temporal_targets = temporal_targets.reshape(batch_size * seq_len * num_agents, -1)
    else:
        temporal_targets = None
    return obs, actions, threat_targets, attack_targets, temporal_targets, int(seq_len), int(batch_size), int(batch_size * seq_len)


def weighted_loss(outputs, args, use_temporal_loss):
    threat_loss = F.mse_loss(outputs["threat_pred"], outputs["threat_targets"])
    attack_loss = F.mse_loss(outputs["attack_pred"], outputs["attack_targets"])
    if use_temporal_loss:
        temporal_loss = F.mse_loss(outputs["temporal_pred"], outputs["temporal_targets"])
    else:
        temporal_loss = torch.zeros((), dtype=torch.float32, device=outputs["threat_pred"].device)
    loss = (
        args.threat_loss_weight * threat_loss
        + args.attack_loss_weight * attack_loss
        + args.temporal_loss_weight * temporal_loss
    )
    return loss, threat_loss, attack_loss, temporal_loss


def forward_losses(model, batch, device, args, use_temporal_loss):
    obs, actions, threat_targets, attack_targets, temporal_targets, seq_len, window_count, step_count = prepare_batch(batch, device)
    temporal_pred, threat_pred, attack_pred = model(obs, actions, seq_len=seq_len, time_offset=0)
    outputs = {
        "temporal_pred": temporal_pred,
        "threat_pred": threat_pred,
        "attack_pred": attack_pred,
        "temporal_targets": temporal_targets,
        "threat_targets": threat_targets,
        "attack_targets": attack_targets,
        "window_count": window_count,
        "step_count": step_count,
    }
    loss, threat_loss, attack_loss, temporal_loss = weighted_loss(outputs, args, use_temporal_loss)
    outputs.update(
        {
            "loss": loss,
            "threat_loss": threat_loss,
            "attack_loss": attack_loss,
            "temporal_loss": temporal_loss,
        }
    )
    return outputs


def metrics_from_outputs(outputs, args):
    return {
        "loss": float(outputs["loss"].item()),
        "threat_loss": float(args.threat_loss_weight * outputs["threat_loss"].item()),
        "attack_loss": float(args.attack_loss_weight * outputs["attack_loss"].item()),
        "temporal_loss": float(args.temporal_loss_weight * outputs["temporal_loss"].item()),
        "raw_threat_loss": float(outputs["threat_loss"].item()),
        "raw_attack_loss": float(outputs["attack_loss"].item()),
        "raw_temporal_loss": float(outputs["temporal_loss"].item()),
        "windows": int(outputs["window_count"]),
        "steps": int(outputs["step_count"]),
    }


def compute_r_value(sse, target_sum, target_square_sum, count):
    if count <= 0:
        return 0.0
    sst = target_square_sum - target_sum * target_sum / count
    if sst <= 1e-12:
        return 0.0
    return 1.0 - sse / sst


def empty_metrics():
    return {
        "loss": 0.0,
        "threat_loss": 0.0,
        "attack_loss": 0.0,
        "temporal_loss": 0.0,
        "raw_threat_loss": 0.0,
        "raw_attack_loss": 0.0,
        "raw_temporal_loss": 0.0,
        "steps": 0,
        "threat_r": 0.0,
        "attack_r": 0.0,
    }


def finalize_metrics(totals):
    steps = max(int(totals["steps"]), 1)
    return {
        "loss": totals["loss"] / steps,
        "threat_loss": totals["threat_loss"] / steps,
        "attack_loss": totals["attack_loss"] / steps,
        "temporal_loss": totals["temporal_loss"] / steps,
        "raw_threat_loss": totals["raw_threat_loss"] / steps,
        "raw_attack_loss": totals["raw_attack_loss"] / steps,
        "raw_temporal_loss": totals["raw_temporal_loss"] / steps,
        "steps": int(totals["steps"]),
        "threat_r": compute_r_value(totals["threat_sse"], totals["threat_sum"], totals["threat_square_sum"], totals["threat_count"]),
        "attack_r": compute_r_value(totals["attack_sse"], totals["attack_sum"], totals["attack_square_sum"], totals["attack_count"]),
    }


def update_totals(totals, outputs, args):
    step_count = int(outputs["step_count"])
    totals["loss"] += outputs["loss"].item() * step_count
    totals["threat_loss"] += args.threat_loss_weight * outputs["threat_loss"].item() * step_count
    totals["attack_loss"] += args.attack_loss_weight * outputs["attack_loss"].item() * step_count
    totals["temporal_loss"] += args.temporal_loss_weight * outputs["temporal_loss"].item() * step_count
    totals["raw_threat_loss"] += outputs["threat_loss"].item() * step_count
    totals["raw_attack_loss"] += outputs["attack_loss"].item() * step_count
    totals["raw_temporal_loss"] += outputs["temporal_loss"].item() * step_count
    totals["steps"] += step_count

    threat_error = outputs["threat_pred"] - outputs["threat_targets"]
    attack_error = outputs["attack_pred"] - outputs["attack_targets"]
    totals["threat_sse"] += torch.sum(threat_error * threat_error).item()
    totals["attack_sse"] += torch.sum(attack_error * attack_error).item()
    totals["threat_sum"] += torch.sum(outputs["threat_targets"]).item()
    totals["attack_sum"] += torch.sum(outputs["attack_targets"]).item()
    totals["threat_square_sum"] += torch.sum(outputs["threat_targets"] * outputs["threat_targets"]).item()
    totals["attack_square_sum"] += torch.sum(outputs["attack_targets"] * outputs["attack_targets"]).item()
    totals["threat_count"] += outputs["threat_targets"].numel()
    totals["attack_count"] += outputs["attack_targets"].numel()


def new_totals():
    return {
        "loss": 0.0,
        "threat_loss": 0.0,
        "attack_loss": 0.0,
        "temporal_loss": 0.0,
        "raw_threat_loss": 0.0,
        "raw_attack_loss": 0.0,
        "raw_temporal_loss": 0.0,
        "steps": 0,
        "threat_sse": 0.0,
        "attack_sse": 0.0,
        "threat_sum": 0.0,
        "attack_sum": 0.0,
        "threat_square_sum": 0.0,
        "attack_square_sum": 0.0,
        "threat_count": 0,
        "attack_count": 0,
    }


def log_update(epoch, epochs, update_index, update_total, metrics, elapsed):
    logging.info(
        f"  [E{epoch:03d}/{epochs:03d} U{update_index:03d}/{update_total:03d}] "
        f"loss={metrics['loss']:.5f} "
        f"| raw(th/a/t)=({metrics['raw_threat_loss']:.5f}/"
        f"{metrics['raw_attack_loss']:.5f}/"
        f"{metrics['raw_temporal_loss']:.5f}) "
        f"| weighted(th/a/t)=({metrics['threat_loss']:.5f}/"
        f"{metrics['attack_loss']:.5f}/"
        f"{metrics['temporal_loss']:.5f}) "
        f"| windows={metrics['windows']} steps={metrics['steps']} "
        f"| {elapsed:.2f}s"
    )


def train_one_epoch(
    model,
    loader,
    device,
    args,
    optimizer,
    optimizers,
    parameter_groups,
    use_temporal_loss,
    epoch,
    update_log_path,
    write_update_header,
):
    model.train()
    totals = new_totals()
    update_count = 0
    update_total = len(loader)
    backbone_params, threat_head_params, attack_head_params = parameter_groups

    for batch in loader:
        update_start = time.time()
        update_count += 1
        outputs = forward_losses(model, batch, device, args, use_temporal_loss)

        if args.train_mode == "split":
            for opt in optimizers.values():
                opt.zero_grad()
            zero_param_grads(backbone_params)
            zero_param_grads(threat_head_params)
            zero_param_grads(attack_head_params)

            temporal_obj = args.temporal_loss_weight * outputs["temporal_loss"]
            threat_obj = args.threat_loss_weight * outputs["threat_loss"]
            attack_obj = args.attack_loss_weight * outputs["attack_loss"]

            backbone_grads = torch.autograd.grad(
                temporal_obj,
                backbone_params,
                retain_graph=True,
                allow_unused=False,
            )
            threat_grads = torch.autograd.grad(
                threat_obj,
                threat_head_params,
                retain_graph=True,
                allow_unused=False,
            )
            attack_grads = torch.autograd.grad(
                attack_obj,
                attack_head_params,
                retain_graph=False,
                allow_unused=False,
            )

            assign_grads(backbone_params, backbone_grads)
            assign_grads(threat_head_params, threat_grads)
            assign_grads(attack_head_params, attack_grads)
            clip_param_grads(backbone_params, args.max_grad_norm)
            clip_param_grads(threat_head_params, args.max_grad_norm)
            clip_param_grads(attack_head_params, args.max_grad_norm)
            optimizers["backbone"].step()
            optimizers["threat_head"].step()
            optimizers["attack_head"].step()
        else:
            optimizer.zero_grad()
            if args.train_mode == "heads_only":
                loss = args.threat_loss_weight * outputs["threat_loss"] + args.attack_loss_weight * outputs["attack_loss"]
            else:
                loss = outputs["loss"]
            loss.backward()
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(optimizer.param_groups[0]["params"], args.max_grad_norm)
            optimizer.step()

        update_totals(totals, outputs, args)
        update_metrics = metrics_from_outputs(outputs, args)
        update_elapsed = time.time() - update_start
        log_update(epoch, args.epochs, update_count, update_total, update_metrics, update_elapsed)
        append_csv_row(
            update_log_path,
            {
                "epoch": epoch,
                "update": update_count,
                "update_total": update_total,
                "windows": update_metrics["windows"],
                "steps": update_metrics["steps"],
                "loss": f"{update_metrics['loss']:.8f}",
                "raw_threat_loss": f"{update_metrics['raw_threat_loss']:.8f}",
                "raw_attack_loss": f"{update_metrics['raw_attack_loss']:.8f}",
                "raw_temporal_loss": f"{update_metrics['raw_temporal_loss']:.8f}",
                "threat_loss": f"{update_metrics['threat_loss']:.8f}",
                "attack_loss": f"{update_metrics['attack_loss']:.8f}",
                "temporal_loss": f"{update_metrics['temporal_loss']:.8f}",
                "time_sec": f"{update_elapsed:.4f}",
            },
            write_header=(write_update_header and update_count == 1),
        )

    metrics = finalize_metrics(totals)
    metrics["updates"] = int(update_count)
    return metrics


def evaluate(model, loader, device, args, use_temporal_loss):
    model.eval()
    totals = new_totals()
    with torch.no_grad():
        for batch in loader:
            outputs = forward_losses(model, batch, device, args, use_temporal_loss)
            update_totals(totals, outputs, args)
    return finalize_metrics(totals)


def append_csv_row(path, row, write_header=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def optimizer_state(args, optimizer, optimizers):
    if args.train_mode == "split":
        return {key: opt.state_dict() for key, opt in optimizers.items()}
    return optimizer.state_dict()


def save_checkpoint(path, model, args, optimizer, optimizers, epoch, val_loss):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer_state(args, optimizer, optimizers),
            "epoch": int(epoch),
            "val_loss": float(val_loss),
            "args": vars(args),
        },
        path,
    )


def build_run_dir(args):
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    return resolve_project_path(args.save_root) / args.experiment_name / f"run-{timestamp}-seed{args.seed}"


def dump_config(path, args, dataset_paths, datasets):
    payload = {
        "args": vars(args),
        "dataset_paths": {key: normalize_path(value) for key, value in dataset_paths.items()},
        "dataset_summary": {
            key: {
                "windows": int(len(dataset)),
                "window_length": int(dataset.window_length),
                "steps": int(dataset.step_count),
                "obs_shape": list(dataset.obs.shape),
                "actions_shape": list(dataset.actions.shape),
            }
            for key, dataset in datasets.items()
        },
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def log_epoch(epoch, args, train_metrics, val_metrics, elapsed):
    logging.info(
        f"[E{epoch:03d}/{args.epochs:03d}] "
        f"train={train_metrics['loss']:.5f} "
        f"| val={val_metrics['loss']:.5f} "
        f"| updates={train_metrics.get('updates', 0)} "
        f"| raw(th/a/t)=({val_metrics['raw_threat_loss']:.5f}/"
        f"{val_metrics['raw_attack_loss']:.5f}/"
        f"{val_metrics['raw_temporal_loss']:.5f}) "
        f"| R(th/a)=({val_metrics['threat_r']:.3f}/{val_metrics['attack_r']:.3f}) "
        f"| {elapsed:.1f}s"
    )


def get_parser():
    parser = argparse.ArgumentParser(description="Train AeroTAF_ATTN on prebuilt fixed-window AeroTAF datasets.")
    parser.add_argument("--dataset-dir", type=str, required=True, help="Processed dataset directory.")
    parser.add_argument("--train-file", type=str, default="train.npz", help="Training split filename.")
    parser.add_argument("--val-file", type=str, default="val.npz", help="Validation split filename.")
    parser.add_argument("--test-file", type=str, default="test.npz", help="Test split filename.")
    parser.add_argument("--experiment-name", type=str, default="AeroTAF-ATTN-Windows-K20-L50-S30", help="Experiment name.")
    parser.add_argument("--save-root", type=str, default="scripts/results/AeroTAF_ATTN", help="Checkpoint root directory.")
    parser.add_argument("--seed", type=int, default=1, help="Random seed.")
    parser.add_argument("--cuda", action="store_true", default=False, help="Use CUDA if available.")
    parser.add_argument("--cuda-device-id", type=int, default=0, help="CUDA device id.")
    parser.add_argument("--n-training-threads", type=int, default=1, help="Torch training thread count.")
    parser.add_argument("--mini-epoch", type=int, default=10, help="Number of parameter updates per epoch; each update uses roughly train_windows / mini_epoch windows.")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs.")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Adam weight decay.")
    parser.add_argument("--max-grad-norm", type=float, default=2.0, help="Gradient clipping max norm.")
    parser.add_argument(
        "--train-mode",
        type=str,
        default="split",
        choices=("heads_only", "joint", "split"),
        help="heads_only trains field heads only; joint trains all losses together; split updates backbone by temporal loss and heads by field losses.",
    )
    parser.add_argument("--threat-loss-weight", type=float, default=1.0, help="Weight for threat regression loss.")
    parser.add_argument("--attack-loss-weight", type=float, default=8.0, help="Weight for attack regression loss.")
    parser.add_argument("--temporal-loss-weight", type=float, default=0.5, help="Weight for temporal feature regression loss.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker count.")
    parser.add_argument("--save-interval", type=int, default=10, help="Latest checkpoint save interval in epochs.")
    parser.add_argument("--log-interval", type=int, default=1, help="Epoch logging interval.")
    parser.add_argument("--num-agents", type=int, default=4, help="Number of ego agents.")
    parser.add_argument("--activation-id", type=int, default=1, help="0:Tanh, 1:ReLU, 2:LeakyReLU, 3:ELU.")
    parser.add_argument("--use-feature-normalization", action="store_true", default=False, help="Use LayerNorm on obs/action inputs.")
    parser.add_argument("--KQ-hidden-size", type=str, default="128 128", help="Spatial/time K/Q MLP hidden sizes.")
    parser.add_argument("--V-hidden-size", type=str, default="128 128", help="Spatial/time V MLP hidden sizes.")
    parser.add_argument("--attn-output-hidden-size", type=str, default="128", help="Temporal attention FFN hidden sizes.")
    parser.add_argument("--field-output-hidden-size", type=str, default="64 32", help="Threat/attack head hidden sizes.")
    parser.add_argument("--num-heads", type=int, default=4, help="Spatial attention head count.")
    parser.add_argument("--time-head-num", type=int, default=4, help="Temporal attention head count.")
    return parser


def main(args):
    parser = get_parser()
    all_args = parser.parse_args(args)
    use_temporal_loss = all_args.train_mode != "heads_only"

    set_seed(all_args.seed)
    torch.set_num_threads(all_args.n_training_threads)
    device = build_device(all_args)

    dataset_dir = resolve_project_path(all_args.dataset_dir)
    dataset_paths = {
        "dataset_dir": dataset_dir,
        "train": dataset_dir / all_args.train_file,
        "val": dataset_dir / all_args.val_file,
        "test": dataset_dir / all_args.test_file,
    }
    for split_name in ("train", "val", "test"):
        if not dataset_paths[split_name].exists():
            raise FileNotFoundError(f"{split_name} split not found: {dataset_paths[split_name]}")

    datasets = {
        "train": AeroTAFWindowStore(dataset_paths["train"], use_temporal_loss=use_temporal_loss),
        "val": AeroTAFWindowStore(dataset_paths["val"], use_temporal_loss=use_temporal_loss),
        "test": AeroTAFWindowStore(dataset_paths["test"], use_temporal_loss=use_temporal_loss),
    }
    all_args.effective_mini_epoch = max(1, min(int(all_args.mini_epoch), len(datasets["train"])))
    all_args.eval_batch_size = max(1, int(np.ceil(len(datasets["train"]) / all_args.effective_mini_epoch)))

    obs_space, act_space = build_spaces(datasets["train"])
    model = PPOAeroTAFATTN(AeroTAFATTNArgs(all_args), obs_space, act_space, device=device)
    optimizer, optimizers, backbone_params, threat_head_params, attack_head_params = build_optimizers(model, all_args)

    loaders = {
        "val": build_eval_loader(datasets["val"], all_args, device),
        "test": build_eval_loader(datasets["test"], all_args, device),
    }

    run_dir = build_run_dir(all_args)
    run_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = run_dir / "AeroTAF_ATTN_best.pt"
    latest_ckpt_path = run_dir / "AeroTAF_ATTN_latest.pt"
    update_log_path = run_dir / "update_log.csv"
    epoch_log_path = run_dir / "epoch_log.csv"
    config_path = run_dir / "config.json"
    test_metrics_path = run_dir / "test_metrics.json"
    dump_config(config_path, all_args, dataset_paths, datasets)

    logging.info("=" * 72)
    logging.info("AeroTAF_ATTN Window Training")
    logging.info("=" * 72)
    logging.info(f"device     : {device}")
    logging.info(f"dataset    : {normalize_path(dataset_dir)}")
    logging.info(f"run dir    : {normalize_path(run_dir)}")
    logging.info(
        f"windows    : train={len(datasets['train'])} val={len(datasets['val'])} test={len(datasets['test'])} "
        f"| length={datasets['train'].window_length}"
    )
    logging.info(
        f"train mode : {all_args.train_mode} | mini_epoch={all_args.effective_mini_epoch} "
        f"| windows/update~{all_args.eval_batch_size} | epochs={all_args.epochs} | lr={all_args.lr:.2e}"
    )
    logging.info(
        f"loss w     : threat={all_args.threat_loss_weight} attack={all_args.attack_loss_weight} temporal={all_args.temporal_loss_weight}"
    )
    if all_args.train_mode == "split":
        logging.info(
            f"params     : backbone={len(backbone_params)} threat_head={len(threat_head_params)} attack_head={len(attack_head_params)}"
        )
    logging.info("-" * 72)

    initial_val_start = time.time()
    best_val_loss = float("inf")
    parameter_groups = (backbone_params, threat_head_params, attack_head_params)
    initial_val_metrics = evaluate(model, loaders["val"], device, all_args, use_temporal_loss)
    best_val_loss = initial_val_metrics["loss"]
    save_checkpoint(best_ckpt_path, model, all_args, optimizer, optimizers, 0, best_val_loss)
    logging.info(
        f"[INIT] val={initial_val_metrics['loss']:.6f} "
        f"| raw(th/a/t)=({initial_val_metrics['raw_threat_loss']:.6f}/"
        f"{initial_val_metrics['raw_attack_loss']:.6f}/"
        f"{initial_val_metrics['raw_temporal_loss']:.6f}) "
        f"| R(th/a)=({initial_val_metrics['threat_r']:.4f}/{initial_val_metrics['attack_r']:.4f}) "
        f"| {time.time() - initial_val_start:.1f}s"
    )

    for epoch in range(1, all_args.epochs + 1):
        start_time = time.time()
        train_loader = build_train_loader(datasets["train"], all_args, device, epoch)
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            device=device,
            args=all_args,
            optimizer=optimizer,
            optimizers=optimizers,
            parameter_groups=parameter_groups,
            use_temporal_loss=use_temporal_loss,
            epoch=epoch,
            update_log_path=update_log_path,
            write_update_header=(epoch == 1),
        )
        val_metrics = evaluate(model, loaders["val"], device, all_args, use_temporal_loss)
        elapsed = time.time() - start_time

        is_best = val_metrics["loss"] < best_val_loss
        if is_best:
            best_val_loss = val_metrics["loss"]
            save_checkpoint(best_ckpt_path, model, all_args, optimizer, optimizers, epoch, best_val_loss)
            logging.info(f"[BEST] epoch={epoch:03d} val={best_val_loss:.6f} -> {normalize_path(best_ckpt_path)}")

        if epoch % all_args.save_interval == 0 or epoch == all_args.epochs:
            save_checkpoint(latest_ckpt_path, model, all_args, optimizer, optimizers, epoch, best_val_loss)

        row = {
            "epoch": epoch,
            "train_loss": f"{train_metrics['loss']:.8f}",
            "train_raw_threat_loss": f"{train_metrics['raw_threat_loss']:.8f}",
            "train_raw_attack_loss": f"{train_metrics['raw_attack_loss']:.8f}",
            "train_raw_temporal_loss": f"{train_metrics['raw_temporal_loss']:.8f}",
            "train_updates": int(train_metrics.get("updates", 0)),
            "val_loss": f"{val_metrics['loss']:.8f}",
            "val_raw_threat_loss": f"{val_metrics['raw_threat_loss']:.8f}",
            "val_raw_attack_loss": f"{val_metrics['raw_attack_loss']:.8f}",
            "val_raw_temporal_loss": f"{val_metrics['raw_temporal_loss']:.8f}",
            "best_val_loss": f"{best_val_loss:.8f}",
            "time_sec": f"{elapsed:.4f}",
        }
        append_csv_row(epoch_log_path, row, write_header=(epoch == 1))

        if epoch % all_args.log_interval == 0 or epoch == 1 or epoch == all_args.epochs:
            log_epoch(epoch, all_args, train_metrics, val_metrics, elapsed)

    if best_ckpt_path.exists():
        checkpoint = torch.load(best_ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        logging.info(f"loaded best checkpoint for final test: {normalize_path(best_ckpt_path)}")

    test_metrics = evaluate(model, loaders["test"], device, all_args, use_temporal_loss)
    with open(test_metrics_path, "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2, ensure_ascii=False)

    logging.info("-" * 72)
    logging.info(
        f"[TEST] loss={test_metrics['loss']:.6f} "
        f"| raw(th/a/t)=({test_metrics['raw_threat_loss']:.6f}/"
        f"{test_metrics['raw_attack_loss']:.6f}/"
        f"{test_metrics['raw_temporal_loss']:.6f}) "
        f"| R(th/a)=({test_metrics['threat_r']:.4f}/{test_metrics['attack_r']:.4f})"
    )
    logging.info(f"test metrics saved: {normalize_path(test_metrics_path)}")
    logging.info("Done.")


if __name__ == "__main__":
    default_args = [
        "--dataset-dir", "datasets/aerotaf/4v4_shoot_mappo_pool_stage1/processed_stage1_K20_L50_S30",
        "--experiment-name", "AeroTAF-ATTN-Windows-K20-L50-S30",
        "--save-root", "scripts/results/AeroTAF_ATTN",
        "--seed", "1",
        "--n-training-threads", "1",
        "--mini-epoch", "10",
        "--epochs", "50",
        "--lr", "3e-5",
        "--weight-decay", "1e-4",
        "--max-grad-norm", "2.0",
        "--train-mode", "split",
        "--threat-loss-weight", "1.0",
        "--attack-loss-weight", "8.0",
        "--temporal-loss-weight", "0.5",
        "--num-workers", "0",
        "--save-interval", "10",
        "--log-interval", "1",
        "--num-agents", "4",
        "--activation-id", "1",
        "--KQ-hidden-size", "128 128",
        "--V-hidden-size", "128 128",
        "--attn-output-hidden-size", "128",
        "--field-output-hidden-size", "64 32",
        "--num-heads", "4",
        "--time-head-num", "4",
        "--use-feature-normalization",
        # "--cuda",
        # "--cuda-device-id", "0",
    ]
    main(sys.argv[1:] if len(sys.argv) > 1 else default_args)
