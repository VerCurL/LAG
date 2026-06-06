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
from scripts.AeroTAF.data.processed_store import ProcessedEpisodeStore
from scripts.AeroTAF.data.windows import AeroTAFWindowDataset


TEMPORAL_BEST_NAME = "AeroTAF_ATTN_temporal_best.pt"
TEMPORAL_LATEST_NAME = "AeroTAF_ATTN_temporal_latest.pt"
THREAT_BEST_NAME = "AeroTAF_ATTN_threat_best.pt"
THREAT_LATEST_NAME = "AeroTAF_ATTN_threat_latest.pt"
ATTACK_BEST_NAME = "AeroTAF_ATTN_attack_best.pt"
ATTACK_LATEST_NAME = "AeroTAF_ATTN_attack_latest.pt"


class AeroTAFATTNArgs:
    def __init__(self, args):
        self.num_agents = args.num_agents
        self.activation_id = args.activation_id
        self.use_feature_normalization = args.use_feature_normalization
        self.num_heads = args.num_heads
        self.time_head_num = args.time_head_num
        self.KQ_hidden_size = args.KQ_hidden_size
        self.V_hidden_size = args.V_hidden_size
        self.AeroTAF_out_hidden_size = args.AeroTAF_out_hidden_size


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
            output_hidden_size=args.AeroTAF_out_hidden_size,
            activation_id=args.activation_id,
            use_feature_normalization=args.use_feature_normalization,
        )
        self.to(device)

    def forward(self, obs, actions, seq_len=None, time_offset=0):
        obs = torch.as_tensor(obs, **self.tpdv)
        actions = torch.as_tensor(actions, **self.tpdv)
        return self.AeroTAF(obs, actions, seq_len=seq_len, time_offset=time_offset)


THREAT_HEAD_PATTERN = "threat_output_module"
ATTACK_HEAD_PATTERN = "attack_output_module"


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


def assign_grads(params, grads):
    for param, grad in zip(params, grads):
        param.grad = None if grad is None else grad.detach()


def zero_param_grads(params):
    for param in params:
        param.grad = None


def clip_param_grads(params, max_grad_norm):
    if max_grad_norm is None or max_grad_norm <= 0 or not params:
        return
    torch.nn.utils.clip_grad_norm_(params, max_grad_norm)


def build_split_optimizers(model, args):
    backbone_params, threat_head_params, attack_head_params = split_parameter_groups(model)
    optimizers = {
        "temporal": torch.optim.Adam(backbone_params, lr=args.lr_temporal, weight_decay=args.weight_decay),
        "threat": torch.optim.Adam(threat_head_params, lr=args.lr_threat, weight_decay=args.weight_decay),
        "attack": torch.optim.Adam(attack_head_params, lr=args.lr_attack, weight_decay=args.weight_decay),
    }
    return optimizers, backbone_params, threat_head_params, attack_head_params


def episode_collate_fn(batch):
    if len(batch) != 1:
        raise ValueError("Current ATTN training expects one window per loader step.")
    return batch[0]


class IndexDataset(Dataset):
    def __init__(self, dataset, indices, type_counts=None):
        self.dataset = dataset
        self.indices = list(indices)
        self.type_counts = dict(type_counts or {})

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        return self.dataset[self.indices[index]]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_spaces(store):
    obs_dim = store.obs.shape[-1]
    act_dim = store.actions.shape[-1]
    obs_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
    act_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(act_dim,), dtype=np.float32)
    return obs_space, act_space


def build_device(args):
    if args.cuda and torch.cuda.is_available():
        if args.cuda_device_id >= 0:
            return torch.device(f"cuda:{args.cuda_device_id}")
        return torch.device("cuda:0")
    return torch.device("cpu")


def normalize_ratios(ratios):
    total = sum(max(float(value), 0.0) for value in ratios.values())
    if total <= 0.0:
        raise ValueError("Window ratios must contain at least one positive value.")
    return {key: max(float(value), 0.0) / total for key, value in ratios.items()}


def group_window_indices(dataset):
    groups = {}
    for idx, window in enumerate(dataset.windows):
        window_type = str(window["window_type"])
        groups.setdefault(window_type, []).append(idx)
    return groups


def sample_indices_by_ratio(dataset, total_samples, ratios, seed):
    groups = group_window_indices(dataset)
    available_types = [key for key, indices in groups.items() if indices]
    if not available_types:
        raise RuntimeError("No windows available for ratio sampling.")

    filtered = {key: ratios.get(key, 0.0) for key in available_types}
    normalized = normalize_ratios(filtered)
    rng = np.random.default_rng(seed)

    counts = {key: int(total_samples * normalized[key]) for key in normalized}
    assigned = sum(counts.values())
    leftovers = int(total_samples) - assigned
    ordered_types = sorted(normalized.items(), key=lambda item: item[1], reverse=True)
    cursor = 0
    while leftovers > 0 and ordered_types:
        key = ordered_types[cursor % len(ordered_types)][0]
        counts[key] += 1
        leftovers -= 1
        cursor += 1

    sampled = []
    for key, count in counts.items():
        if count <= 0:
            continue
        indices = np.asarray(groups[key], dtype=np.int64)
        replace = len(indices) < count
        chosen = rng.choice(indices, size=count, replace=replace)
        sampled.extend(chosen.tolist())

    rng.shuffle(sampled)
    sampled_type_counts = {}
    for index in sampled:
        window_type = str(dataset.windows[int(index)]["window_type"])
        sampled_type_counts[window_type] = sampled_type_counts.get(window_type, 0) + 1
    return sampled, sampled_type_counts


def count_window_types(dataset, indices):
    counts = {}
    windows = getattr(dataset, "windows", None)
    if windows is None and hasattr(dataset, "dataset"):
        windows = getattr(dataset.dataset, "windows", None)
    if windows is None:
        return counts
    for index in indices:
        window_type = str(windows[int(index)]["window_type"])
        counts[window_type] = counts.get(window_type, 0) + 1
    return counts


def build_loader(dataset, shuffle, num_workers, device):
    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=episode_collate_fn,
        drop_last=False,
    )


def prepare_episode_batch(
    obs,
    actions,
    threat_targets,
    attack_targets,
    temporal_targets,
    sample_weight,
    sample_multi_hot,
    seq_len,
    time_offset,
    *unused_fields,
    device,
):
    seq_len = int(seq_len)
    time_offset = int(time_offset)

    obs = torch.as_tensor(obs, dtype=torch.float32, device=device).reshape(seq_len, obs.shape[1], -1)
    actions = torch.as_tensor(actions, dtype=torch.float32, device=device).reshape(seq_len, actions.shape[1], -1)
    threat_targets = torch.as_tensor(threat_targets, dtype=torch.float32, device=device).reshape(seq_len, -1)
    attack_targets = torch.as_tensor(attack_targets, dtype=torch.float32, device=device).reshape(seq_len, -1)
    sample_weight = torch.as_tensor(sample_weight, dtype=torch.float32, device=device).reshape(seq_len, -1)
    sample_multi_hot = torch.as_tensor(sample_multi_hot, dtype=torch.float32, device=device).reshape(seq_len, -1)

    if temporal_targets is not None:
        temporal_targets = torch.as_tensor(temporal_targets, dtype=torch.float32, device=device).reshape(
            seq_len,
            temporal_targets.shape[1],
            -1,
        )
        temporal_targets = temporal_targets.reshape(seq_len * temporal_targets.shape[1], -1)

    obs = obs.reshape(seq_len * obs.shape[1], -1)
    actions = actions.reshape(seq_len * actions.shape[1], -1)
    return (
        obs,
        actions,
        threat_targets,
        attack_targets,
        temporal_targets,
        sample_weight,
        sample_multi_hot,
        seq_len,
        time_offset,
    )


def weighted_time_mse(pred, target, sample_weight):
    loss_raw = (pred - target).pow(2)
    weighted = loss_raw * sample_weight
    denom = torch.clamp(sample_weight.sum(), min=1e-6)
    return weighted.sum() / denom


def weighted_temporal_mse(pred, target, sample_weight, num_agents):
    expanded_weight = sample_weight.repeat_interleave(int(num_agents), dim=0)
    loss_raw = (pred - target).pow(2)
    weighted = loss_raw * expanded_weight
    denom = torch.clamp(expanded_weight.sum() * pred.shape[-1], min=1e-6)
    return weighted.sum() / denom


def compute_r_value(sse, target_sum, target_square_sum, count):
    if count <= 0:
        return 0.0
    sst = target_square_sum - target_sum * target_sum / count
    if sst <= 1e-12:
        return 0.0
    return 1.0 - sse / sst


def run_forward_losses(model, batch, device, num_agents):
    if len(batch) < 9:
        raise ValueError(f"Unexpected batch format with {len(batch)} fields.")

    (
        obs,
        actions,
        threat_targets,
        attack_targets,
        temporal_targets,
        sample_weight,
        sample_multi_hot,
        seq_len,
        time_offset,
        *extra_fields,
    ) = batch

    (
        obs,
        actions,
        threat_targets,
        attack_targets,
        temporal_targets,
        sample_weight,
        sample_multi_hot,
        seq_len,
        time_offset,
    ) = prepare_episode_batch(
        obs,
        actions,
        threat_targets,
        attack_targets,
        temporal_targets,
        sample_weight,
        sample_multi_hot,
        seq_len,
        time_offset,
        *extra_fields,
        device=device,
    )

    temporal_pred, threat_pred, attack_pred = model(obs, actions, seq_len=seq_len, time_offset=time_offset)
    temporal_loss = weighted_temporal_mse(temporal_pred, temporal_targets, sample_weight, num_agents)
    threat_loss = weighted_time_mse(threat_pred, threat_targets, sample_weight)
    attack_loss = weighted_time_mse(attack_pred, attack_targets, sample_weight)

    return {
        "temporal_pred": temporal_pred,
        "threat_pred": threat_pred,
        "attack_pred": attack_pred,
        "temporal_targets": temporal_targets,
        "threat_targets": threat_targets,
        "attack_targets": attack_targets,
        "sample_weight": sample_weight,
        "sample_multi_hot": sample_multi_hot,
        "seq_len": seq_len,
        "time_offset": time_offset,
        "temporal_loss": temporal_loss,
        "threat_loss": threat_loss,
        "attack_loss": attack_loss,
    }


def evaluate(model, data_loader, device, args, return_stats=False):
    model.eval()
    total_loss = 0.0
    total_temporal_loss = 0.0
    total_threat_loss = 0.0
    total_attack_loss = 0.0
    total_steps = 0
    threat_sse = 0.0
    attack_sse = 0.0
    threat_sum = 0.0
    attack_sum = 0.0
    threat_square_sum = 0.0
    attack_square_sum = 0.0
    threat_count = 0
    attack_count = 0

    with torch.no_grad():
        for batch in data_loader:
            outputs = run_forward_losses(model, batch, device=device, num_agents=args.num_agents)
            loss = (
                args.temporal_loss_weight * outputs["temporal_loss"]
                + args.threat_loss_weight * outputs["threat_loss"]
                + args.attack_loss_weight * outputs["attack_loss"]
            )

            seq_len = int(outputs["seq_len"])
            total_loss += loss.item() * seq_len
            total_temporal_loss += outputs["temporal_loss"].item() * seq_len
            total_threat_loss += outputs["threat_loss"].item() * seq_len
            total_attack_loss += outputs["attack_loss"].item() * seq_len
            total_steps += seq_len

            threat_error = outputs["threat_pred"] - outputs["threat_targets"]
            attack_error = outputs["attack_pred"] - outputs["attack_targets"]
            threat_sse += torch.sum(threat_error * threat_error).item()
            attack_sse += torch.sum(attack_error * attack_error).item()
            threat_sum += torch.sum(outputs["threat_targets"]).item()
            attack_sum += torch.sum(outputs["attack_targets"]).item()
            threat_square_sum += torch.sum(outputs["threat_targets"] * outputs["threat_targets"]).item()
            attack_square_sum += torch.sum(outputs["attack_targets"] * outputs["attack_targets"]).item()
            threat_count += outputs["threat_targets"].numel()
            attack_count += outputs["attack_targets"].numel()

    metrics = {
        "loss": total_loss / max(total_steps, 1),
        "temporal_loss": total_temporal_loss / max(total_steps, 1),
        "threat_loss": total_threat_loss / max(total_steps, 1),
        "attack_loss": total_attack_loss / max(total_steps, 1),
        "steps": int(total_steps),
        "threat_r": compute_r_value(threat_sse, threat_sum, threat_square_sum, threat_count),
        "attack_r": compute_r_value(attack_sse, attack_sum, attack_square_sum, attack_count),
    }
    if return_stats:
        metrics.update(
            {
                "samples": int(total_steps),
                "threat_sse": float(threat_sse),
                "attack_sse": float(attack_sse),
                "threat_sum": float(threat_sum),
                "attack_sum": float(attack_sum),
                "threat_square_sum": float(threat_square_sum),
                "attack_square_sum": float(attack_square_sum),
                "threat_count": int(threat_count),
                "attack_count": int(attack_count),
            }
        )
    return metrics


def train_one_epoch(model, data_loader, device, args, optimizers, backbone_params, threat_head_params, attack_head_params):
    model.train()
    total_loss = 0.0
    total_temporal_loss = 0.0
    total_threat_loss = 0.0
    total_attack_loss = 0.0
    total_steps = 0

    for batch in data_loader:
        outputs = run_forward_losses(model, batch, device=device, num_agents=args.num_agents)
        temporal_obj = args.temporal_loss_weight * outputs["temporal_loss"]
        threat_obj = args.threat_loss_weight * outputs["threat_loss"]
        attack_obj = args.attack_loss_weight * outputs["attack_loss"]

        for optimizer in optimizers.values():
            optimizer.zero_grad()
        zero_param_grads(backbone_params)
        zero_param_grads(threat_head_params)
        zero_param_grads(attack_head_params)

        retain_for_temporal = bool(threat_head_params or attack_head_params)
        backbone_grads = torch.autograd.grad(
            temporal_obj,
            backbone_params,
            retain_graph=retain_for_temporal,
            allow_unused=False,
        )
        threat_grads = torch.autograd.grad(
            threat_obj,
            threat_head_params,
            retain_graph=bool(attack_head_params),
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

        optimizers["temporal"].step()
        optimizers["threat"].step()
        optimizers["attack"].step()

        seq_len = int(outputs["seq_len"])
        total_loss += (temporal_obj.item() + threat_obj.item() + attack_obj.item()) * seq_len
        total_temporal_loss += outputs["temporal_loss"].item() * seq_len
        total_threat_loss += outputs["threat_loss"].item() * seq_len
        total_attack_loss += outputs["attack_loss"].item() * seq_len
        total_steps += seq_len

    return {
        "loss": total_loss / max(total_steps, 1),
        "temporal_loss": total_temporal_loss / max(total_steps, 1),
        "threat_loss": total_threat_loss / max(total_steps, 1),
        "attack_loss": total_attack_loss / max(total_steps, 1),
        "steps": int(total_steps),
    }


def merge_train_metrics(metrics_list):
    total_steps = sum(int(item["steps"]) for item in metrics_list)
    if total_steps <= 0:
        merged = {
            "loss": 0.0,
            "temporal_loss": 0.0,
            "threat_loss": 0.0,
            "attack_loss": 0.0,
            "steps": 0,
        }
        if any("threat_r" in item for item in metrics_list):
            merged["threat_r"] = 0.0
            merged["attack_r"] = 0.0
        return merged

    merged = {
        "loss": sum(float(item["loss"]) * int(item["steps"]) for item in metrics_list) / total_steps,
        "temporal_loss": sum(float(item["temporal_loss"]) * int(item["steps"]) for item in metrics_list) / total_steps,
        "threat_loss": sum(float(item["threat_loss"]) * int(item["steps"]) for item in metrics_list) / total_steps,
        "attack_loss": sum(float(item["attack_loss"]) * int(item["steps"]) for item in metrics_list) / total_steps,
        "steps": int(total_steps),
    }
    if any("threat_r" in item for item in metrics_list):
        merged["threat_r"] = (
            sum(float(item.get("threat_r", 0.0)) * int(item["steps"]) for item in metrics_list) / total_steps
        )
        merged["attack_r"] = (
            sum(float(item.get("attack_r", 0.0)) * int(item["steps"]) for item in metrics_list) / total_steps
        )
    return merged


def append_csv_row(path, row, write_header=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(row.keys())
    if write_header and not path.exists():
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerow(row)
        return
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writerow(row)


def build_run_dir(args):
    save_root = resolve_project_path(args.save_root)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_name = f"run-{timestamp}-seed{args.seed}"
    return save_root / args.experiment_name / run_name


def dump_config(path, args, dataset_paths, stores, datasets):
    payload = {
        "args": vars(args),
        "dataset_paths": {key: normalize_path(value) for key, value in dataset_paths.items()},
        "dataset_summary": {
            "train_episodes": int(len(stores["train"])),
            "val_episodes": int(len(stores["val"])),
            "test_episodes": int(len(stores["test"])),
            "train_steps": int(stores["train"].obs.shape[0]),
            "val_steps": int(stores["val"].obs.shape[0]),
            "test_steps": int(stores["test"].obs.shape[0]),
        },
        "window_summary": {
            split_name: int(len(dataset))
            for split_name, dataset in datasets.items()
        },
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def get_temporal_state_dict(model):
    state = {}
    base = model.AeroTAF
    if getattr(base, "_use_feature_normalization", False):
        state["obs_feature_norm"] = base.obs_feature_norm.state_dict()
        state["act_feature_norm"] = base.act_feature_norm.state_dict()
    layer = base.AeroTAF
    state["K_module"] = layer.K_module.state_dict()
    state["Q_module"] = layer.Q_module.state_dict()
    state["V_module"] = layer.V_module.state_dict()
    state["attn_output_module"] = layer.attn_output_module.state_dict()
    state["norm"] = layer.norm.state_dict()
    state["time_K_module"] = layer.time_K_module.state_dict()
    state["time_Q_module"] = layer.time_Q_module.state_dict()
    state["time_V_module"] = layer.time_V_module.state_dict()
    state["time_attn_output_module"] = layer.time_attn_output_module.state_dict()
    state["time_attn_norm"] = layer.time_attn_norm.state_dict()
    state["time_ffn_module"] = layer.time_ffn_module.state_dict()
    state["time_ffn_norm"] = layer.time_ffn_norm.state_dict()
    state["rope_inv_freq"] = layer.rope_inv_freq.detach().cpu()
    return state


def get_threat_state_dict(model):
    return model.AeroTAF.AeroTAF.threat_output_module.state_dict()


def get_attack_state_dict(model):
    return model.AeroTAF.AeroTAF.attack_output_module.state_dict()


def save_component_checkpoints(run_dir, model, optimizers, epoch, score, args, best=False, mini_index=0):
    run_dir.mkdir(parents=True, exist_ok=True)
    suffix = "best" if best else "latest"
    temporal_path = run_dir / (TEMPORAL_BEST_NAME if best else TEMPORAL_LATEST_NAME)
    threat_path = run_dir / (THREAT_BEST_NAME if best else THREAT_LATEST_NAME)
    attack_path = run_dir / (ATTACK_BEST_NAME if best else ATTACK_LATEST_NAME)

    torch.save(
        {
            "component": "temporal",
            "epoch": epoch,
            "mini_index": int(mini_index),
            "score": float(score),
            "args": vars(args),
            "state_dict": get_temporal_state_dict(model),
            "optimizer_state_dict": optimizers["temporal"].state_dict(),
        },
        temporal_path,
    )
    torch.save(
        {
            "component": "threat",
            "epoch": epoch,
            "mini_index": int(mini_index),
            "score": float(score),
            "args": vars(args),
            "state_dict": get_threat_state_dict(model),
            "optimizer_state_dict": optimizers["threat"].state_dict(),
        },
        threat_path,
    )
    torch.save(
        {
            "component": "attack",
            "epoch": epoch,
            "mini_index": int(mini_index),
            "score": float(score),
            "args": vars(args),
            "state_dict": get_attack_state_dict(model),
            "optimizer_state_dict": optimizers["attack"].state_dict(),
        },
        attack_path,
    )
    return {
        "temporal": temporal_path,
        "threat": threat_path,
        "attack": attack_path,
        "suffix": suffix,
    }


def window_type_counts(dataset):
    counts = {}
    if hasattr(dataset, "type_counts") and dataset.type_counts:
        for window_type, count in dataset.type_counts.items():
            counts[window_type] = counts.get(window_type, 0) + int(count)
        return counts
    for window in dataset.windows:
        counts[window["window_type"]] = counts.get(window["window_type"], 0) + 1
    return counts


def format_window_ratio_string(args):
    return (
        f"event={args.train_ratio_event:.2f} "
        f"high_change={args.train_ratio_high_change:.2f} "
        f"high_field={args.train_ratio_high_field:.2f} "
        f"background={args.train_ratio_background:.2f}"
    )


def format_type_counts(counts):
    ordered = ("event", "high_change", "high_field", "background")
    return " ".join(f"{name}={int(counts.get(name, 0))}" for name in ordered)


def sum_type_counts(type_counts_list):
    total = {}
    for counts in type_counts_list:
        for window_type, count in counts.items():
            total[window_type] = total.get(window_type, 0) + int(count)
    return total


def format_avg_type_counts(counts, divisor):
    ordered = ("event", "high_change", "high_field", "background")
    divisor = max(int(divisor), 1)
    return " ".join(f"{name}={float(counts.get(name, 0)) / divisor:.1f}" for name in ordered)


def build_epoch_mini_datasets(dataset, args, epoch):
    total_samples = len(dataset)
    ratios = {
        "event": args.train_ratio_event,
        "high_change": args.train_ratio_high_change,
        "high_field": args.train_ratio_high_field,
        "background": args.train_ratio_background,
    }
    mini_windows = max(1, int(args.mini_windows))
    mini_sizes = [len(chunk) for chunk in np.array_split(np.arange(total_samples, dtype=np.int64), mini_windows)]
    mini_datasets = []
    sampled_counts = []
    mini_type_counts = []

    for mini_index, mini_size in enumerate(mini_sizes, start=1):
        if mini_size <= 0:
            continue
        indices, type_counts = sample_indices_by_ratio(
            dataset,
            total_samples=int(mini_size),
            ratios=ratios,
            seed=args.seed + epoch * 7919 + mini_index * 101,
        )
        mini_datasets.append(IndexDataset(dataset, indices, type_counts=type_counts))
        sampled_counts.append(len(indices))
        mini_type_counts.append(type_counts)

    return mini_datasets, sampled_counts, mini_type_counts


def build_ordered_mini_datasets(dataset, mini_windows):
    indices = np.arange(len(dataset), dtype=np.int64)
    mini_datasets = []
    mini_counts = []
    mini_type_counts = []
    for chunk in np.array_split(indices, max(1, int(mini_windows))):
        if len(chunk) <= 0:
            continue
        chunk_indices = chunk.tolist()
        type_counts = count_window_types(dataset, chunk_indices)
        mini_datasets.append(IndexDataset(dataset, chunk_indices, type_counts=type_counts))
        mini_counts.append(len(chunk_indices))
        mini_type_counts.append(type_counts)
    return mini_datasets, mini_counts, mini_type_counts


def log_epoch_mini_plan(
    epoch,
    epochs,
    mini_sample_counts,
    mini_type_counts,
    val_natural_counts,
    val_key_counts,
):
    total_counts = sum_type_counts(mini_type_counts)
    min_size = min(mini_sample_counts) if mini_sample_counts else 0
    max_size = max(mini_sample_counts) if mini_sample_counts else 0
    val_nat_min = min(val_natural_counts) if val_natural_counts else 0
    val_nat_max = max(val_natural_counts) if val_natural_counts else 0
    val_key_min = min(val_key_counts) if val_key_counts else 0
    val_key_max = max(val_key_counts) if val_key_counts else 0
    logging.info(
        f"[epoch {epoch:03d}/{epochs:03d}] mini plan: count={len(mini_sample_counts)} "
        f"total_windows={sum(mini_sample_counts)} "
        f"train_size={min_size}-{max_size} "
        f"val_nat_size={val_nat_min}-{val_nat_max} "
        f"val_key_size={val_key_min}-{val_key_max} "
        f"| actual[{format_type_counts(total_counts)}] "
        f"| avg/mini[{format_avg_type_counts(total_counts, len(mini_sample_counts))}]"
    )


def composite_val_score(natural_metrics, key_metrics, args):
    return (
        float(args.val_natural_score_weight) * float(natural_metrics["loss"])
        + float(args.val_key_score_weight) * float(key_metrics["loss"])
    )


def get_parser():
    parser = argparse.ArgumentParser(description="Train AeroTAF_ATTN with split optimization and ratio-based window sampling.")
    parser.add_argument("--dataset-dir", type=str, required=True, help="Processed dataset directory.")
    parser.add_argument("--train-file", type=str, default="train.npz", help="Training split filename.")
    parser.add_argument("--val-file", type=str, default="val.npz", help="Validation split filename.")
    parser.add_argument("--test-file", type=str, default="test.npz", help="Test split filename.")
    parser.add_argument("--experiment-name", type=str, default="AeroTAF-ATTN-Split", help="Experiment name.")
    parser.add_argument("--save-root", type=str, default="scripts/results/AeroTAF_ATTN", help="Checkpoint root directory.")
    parser.add_argument("--seed", type=int, default=1, help="Random seed.")
    parser.add_argument("--cuda", action="store_true", default=False, help="Use CUDA if available.")
    parser.add_argument("--cuda-device-id", type=int, default=0, help="CUDA device id.")
    parser.add_argument("--n-training-threads", type=int, default=1, help="Torch training thread count.")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs.")
    parser.add_argument("--lr-temporal", type=float, default=3e-5, help="Learning rate for temporal backbone.")
    parser.add_argument("--lr-threat", type=float, default=1e-4, help="Learning rate for threat head.")
    parser.add_argument("--lr-attack", type=float, default=1e-4, help="Learning rate for attack head.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Adam weight decay.")
    parser.add_argument("--max-grad-norm", type=float, default=2.0, help="Gradient clipping max norm.")
    parser.add_argument("--temporal-loss-weight", type=float, default=0.5, help="Weight for temporal feature regression loss.")
    parser.add_argument("--threat-loss-weight", type=float, default=1.0, help="Weight for threat regression loss.")
    parser.add_argument("--attack-loss-weight", type=float, default=8.0, help="Weight for attack regression loss.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker count.")
    parser.add_argument("--log-interval", type=int, default=1, help="Epoch logging interval.")

    parser.add_argument("--chunk-length", type=int, default=50, help="Training/eval window length.")
    parser.add_argument("--key-stride", type=int, default=5, help="Stride used to subsample key-centered windows.")
    parser.add_argument("--background-stride", type=int, default=50, help="Stride used for background windows and natural eval windows.")
    parser.add_argument("--mini-windows", type=int, default=10, help="Split each epoch into this many window subsets; each subset is sampled by ratio and trained once.")
    parser.add_argument("--train-ratio-event", type=float, default=0.50, help="Sampling ratio for event windows.")
    parser.add_argument("--train-ratio-high-change", type=float, default=0.20, help="Sampling ratio for high-change windows.")
    parser.add_argument("--train-ratio-high-field", type=float, default=0.20, help="Sampling ratio for high-field windows.")
    parser.add_argument("--train-ratio-background", type=float, default=0.10, help="Sampling ratio for background windows.")
    parser.add_argument("--val-natural-score-weight", type=float, default=0.40, help="Composite best-score weight for natural validation loss.")
    parser.add_argument("--val-key-score-weight", type=float, default=0.60, help="Composite best-score weight for key-window validation loss.")

    parser.add_argument("--num-agents", type=int, default=4, help="Number of ego agents.")
    parser.add_argument("--activation-id", type=int, default=1, help="0:Tanh, 1:ReLU, 2:LeakyReLU, 3:ELU.")
    parser.add_argument("--use-feature-normalization", action="store_true", default=False, help="Use LayerNorm on obs/action inputs.")
    parser.add_argument("--KQ-hidden-size", type=str, default="128 128", help="Spatial/time K/Q MLP hidden sizes.")
    parser.add_argument("--V-hidden-size", type=str, default="128 128", help="Spatial/time V MLP hidden sizes.")
    parser.add_argument("--AeroTAF-out-hidden-size", type=str, default="128", help="Threat/attack head hidden sizes.")
    parser.add_argument("--num-heads", type=int, default=4, help="Spatial attention head count.")
    parser.add_argument("--time-head-num", type=int, default=4, help="Temporal attention head count.")
    return parser


def main(argv):
    parser = get_parser()
    args = parser.parse_args(argv)

    if args.mini_windows < 1:
        raise ValueError("--mini-windows must be >= 1.")

    dataset_dir = resolve_project_path(args.dataset_dir)
    dataset_paths = {
        "dataset_dir": dataset_dir,
        "train": dataset_dir / args.train_file,
        "val": dataset_dir / args.val_file,
        "test": dataset_dir / args.test_file,
    }
    for split_name in ("train", "val", "test"):
        if not dataset_paths[split_name].exists():
            raise FileNotFoundError(f"{split_name} split not found: {dataset_paths[split_name]}")

    set_seed(args.seed)
    torch.set_num_threads(args.n_training_threads)
    device = build_device(args)

    stores = {
        "train": ProcessedEpisodeStore(dataset_paths["train"], require_temporal_targets=True),
        "val": ProcessedEpisodeStore(dataset_paths["val"], require_temporal_targets=True),
        "test": ProcessedEpisodeStore(dataset_paths["test"], require_temporal_targets=True),
    }
    obs_space, act_space = build_spaces(stores["train"])
    model = PPOAeroTAFATTN(AeroTAFATTNArgs(args), obs_space, act_space, device=device)

    optimizers, backbone_params, threat_head_params, attack_head_params = build_split_optimizers(model, args)

    datasets = {
        "train_priority": AeroTAFWindowDataset(stores["train"], args.chunk_length, args.key_stride, args.background_stride, mode="priority"),
        "val_natural": AeroTAFWindowDataset(stores["val"], args.chunk_length, args.key_stride, args.background_stride, mode="natural"),
        "val_key": AeroTAFWindowDataset(stores["val"], args.chunk_length, args.key_stride, args.background_stride, mode="priority"),
        "test_natural": AeroTAFWindowDataset(stores["test"], args.chunk_length, args.key_stride, args.background_stride, mode="natural"),
        "test_key": AeroTAFWindowDataset(stores["test"], args.chunk_length, args.key_stride, args.background_stride, mode="priority"),
    }

    test_natural_loader = build_loader(datasets["test_natural"], False, args.num_workers, device)
    test_key_loader = build_loader(datasets["test_key"], False, args.num_workers, device)

    run_dir = build_run_dir(args)
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / "config.json"
    train_log_path = run_dir / "train_log.csv"
    summary_path = run_dir / "summary_metrics.json"
    dump_config(config_path, args, dataset_paths, stores, datasets)

    logging.info("")
    logging.info("=" * 72)
    logging.info("AeroTAF_ATTN Split Training")
    logging.info("=" * 72)
    logging.info(f"device        : {device}")
    logging.info(f"dataset       : {normalize_path(dataset_dir)}")
    logging.info(f"run dir       : {normalize_path(run_dir)}")
    logging.info(
        f"train data    : episodes={len(stores['train'])} steps={stores['train'].obs.shape[0]} "
        f"windows(priority)={len(datasets['train_priority'])}"
    )
    logging.info(
        f"val/test      : val_nat={len(datasets['val_natural'])} val_key={len(datasets['val_key'])} "
        f"test_nat={len(datasets['test_natural'])} test_key={len(datasets['test_key'])}"
    )
    logging.info(
        f"window config : length={args.chunk_length} key_stride={args.key_stride} "
        f"bg_stride={args.background_stride}"
    )
    logging.info(f"sampling mix  : {format_window_ratio_string(args)}")
    logging.info(
        f"loss weights  : temporal={args.temporal_loss_weight} threat={args.threat_loss_weight} "
        f"attack={args.attack_loss_weight}"
    )
    logging.info(
        f"lr split      : temporal={args.lr_temporal:.2e} threat={args.lr_threat:.2e} attack={args.lr_attack:.2e}"
    )
    logging.info(
        f"train windows : total={len(datasets['train_priority'])} mini_windows={args.mini_windows}"
    )
    logging.info(f"train win cnt : {window_type_counts(datasets['train_priority'])}")
    logging.info("-" * 72)

    best_score = float("inf")
    best_epoch = 0
    best_mini = 0

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        mini_datasets, mini_sample_counts, mini_type_counts = build_epoch_mini_datasets(datasets["train_priority"], args, epoch)
        val_natural_mini_datasets, val_natural_counts, _ = build_ordered_mini_datasets(
            datasets["val_natural"],
            len(mini_datasets),
        )
        val_key_mini_datasets, val_key_counts, _ = build_ordered_mini_datasets(
            datasets["val_key"],
            len(mini_datasets),
        )
        effective_mini_count = min(len(mini_datasets), len(val_natural_mini_datasets), len(val_key_mini_datasets))
        mini_datasets = mini_datasets[:effective_mini_count]
        mini_sample_counts = mini_sample_counts[:effective_mini_count]
        mini_type_counts = mini_type_counts[:effective_mini_count]
        val_natural_mini_datasets = val_natural_mini_datasets[:effective_mini_count]
        val_key_mini_datasets = val_key_mini_datasets[:effective_mini_count]
        val_natural_counts = val_natural_counts[:effective_mini_count]
        val_key_counts = val_key_counts[:effective_mini_count]

        log_epoch_mini_plan(
            epoch,
            args.epochs,
            mini_sample_counts,
            mini_type_counts,
            val_natural_counts,
            val_key_counts,
        )

        train_mini_metrics = []
        val_natural_mini_metrics = []
        val_key_mini_metrics = []
        mini_scores = []

        for mini_index, train_dataset in enumerate(mini_datasets, start=1):
            mini_start = time.time()
            train_loader = build_loader(
                train_dataset,
                shuffle=True,
                num_workers=args.num_workers,
                device=device,
            )
            mini_metrics = train_one_epoch(
                model=model,
                data_loader=train_loader,
                device=device,
                args=args,
                optimizers=optimizers,
                backbone_params=backbone_params,
                threat_head_params=threat_head_params,
                attack_head_params=attack_head_params,
            )
            train_mini_metrics.append(mini_metrics)

            val_natural_loader = build_loader(
                val_natural_mini_datasets[mini_index - 1],
                shuffle=False,
                num_workers=args.num_workers,
                device=device,
            )
            val_key_loader = build_loader(
                val_key_mini_datasets[mini_index - 1],
                shuffle=False,
                num_workers=args.num_workers,
                device=device,
            )
            val_natural_metrics = evaluate(model, val_natural_loader, device, args)
            val_key_metrics = evaluate(model, val_key_loader, device, args)
            mini_score = composite_val_score(val_natural_metrics, val_key_metrics, args)
            val_natural_mini_metrics.append(val_natural_metrics)
            val_key_mini_metrics.append(val_key_metrics)
            mini_scores.append(mini_score)

            if mini_score < best_score:
                best_score = mini_score
                best_epoch = epoch
                best_mini = mini_index
                saved = save_component_checkpoints(
                    run_dir,
                    model,
                    optimizers,
                    epoch,
                    mini_score,
                    args,
                    best=True,
                    mini_index=mini_index,
                )
                logging.info(
                    f"  [BEST] epoch={epoch:03d} mini={mini_index:03d} score={mini_score:.6f} "
                    f"| temporal={saved['temporal'].name} threat={saved['threat'].name} attack={saved['attack'].name}"
                )

            logging.info(
                f"  [mini {mini_index:03d}/{len(mini_datasets):03d}] "
                f"loss={mini_metrics['loss']:.4f} "
                f"| temporal={mini_metrics['temporal_loss']:.4f} "
                f"| threat={mini_metrics['threat_loss']:.4f} "
                f"| attack={mini_metrics['attack_loss']:.4f} "
                f"| val_nat={val_natural_metrics['loss']:.4f} "
                f"| val_key={val_key_metrics['loss']:.4f} "
                f"| score={mini_score:.4f} "
                f"| windows={mini_sample_counts[mini_index - 1]} "
                f"| val=({val_natural_counts[mini_index - 1]}/{val_key_counts[mini_index - 1]}) "
                f"| {format_type_counts(mini_type_counts[mini_index - 1])} "
                f"| {time.time() - mini_start:.1f}s"
            )

        train_metrics = merge_train_metrics(train_mini_metrics)
        val_natural_metrics = merge_train_metrics(val_natural_mini_metrics)
        val_key_metrics = merge_train_metrics(val_key_mini_metrics)
        score = min(mini_scores) if mini_scores else float("inf")
        sampled_count = int(sum(mini_sample_counts))
        epoch_type_counts = sum_type_counts(mini_type_counts)
        elapsed = time.time() - epoch_start

        row = {
            "epoch": epoch,
            "mini_windows": len(mini_sample_counts),
            "train_windows": sampled_count,
            "mini_window_size_mean": f"{(sampled_count / max(len(mini_sample_counts), 1)):.2f}",
            "train_event_windows": int(epoch_type_counts.get("event", 0)),
            "train_high_change_windows": int(epoch_type_counts.get("high_change", 0)),
            "train_high_field_windows": int(epoch_type_counts.get("high_field", 0)),
            "train_background_windows": int(epoch_type_counts.get("background", 0)),
            "train_loss": f"{train_metrics['loss']:.8f}",
            "train_temporal_loss": f"{train_metrics['temporal_loss']:.8f}",
            "train_threat_loss": f"{train_metrics['threat_loss']:.8f}",
            "train_attack_loss": f"{train_metrics['attack_loss']:.8f}",
            "val_natural_loss": f"{val_natural_metrics['loss']:.8f}",
            "val_natural_temporal_loss": f"{val_natural_metrics['temporal_loss']:.8f}",
            "val_natural_threat_loss": f"{val_natural_metrics['threat_loss']:.8f}",
            "val_natural_attack_loss": f"{val_natural_metrics['attack_loss']:.8f}",
            "val_key_loss": f"{val_key_metrics['loss']:.8f}",
            "val_key_temporal_loss": f"{val_key_metrics['temporal_loss']:.8f}",
            "val_key_threat_loss": f"{val_key_metrics['threat_loss']:.8f}",
            "val_key_attack_loss": f"{val_key_metrics['attack_loss']:.8f}",
            "val_key_threat_r": f"{val_key_metrics['threat_r']:.8f}",
            "val_key_attack_r": f"{val_key_metrics['attack_r']:.8f}",
            "score": f"{score:.8f}",
            "best_score": f"{best_score:.8f}",
            "best_epoch": int(best_epoch),
            "best_mini": int(best_mini),
            "time_sec": f"{elapsed:.4f}",
        }
        append_csv_row(train_log_path, row, write_header=(epoch == 1))

        saved = save_component_checkpoints(
            run_dir,
            model,
            optimizers,
            epoch,
            score,
            args,
            best=False,
            mini_index=0,
        )
        logging.info(
            f"[SAVE] epoch={epoch:03d} latest "
            f"| temporal={saved['temporal'].name} threat={saved['threat'].name} attack={saved['attack'].name}"
        )

        if epoch % args.log_interval == 0 or epoch == 1 or epoch == args.epochs:
            logging.info(
                f"[E{epoch:03d}] "
                f"train={train_metrics['loss']:.4f} "
                f"| val_nat={val_natural_metrics['loss']:.4f} "
                f"| val_key={val_key_metrics['loss']:.4f} "
                f"| keyR(t/a)=({val_key_metrics['threat_r']:.3f}/{val_key_metrics['attack_r']:.3f}) "
                f"| epoch_best={score:.4f} "
                f"| best={best_score:.4f}@E{best_epoch:03d}M{best_mini:03d} "
                f"| windows={sampled_count}/{len(mini_sample_counts)}mini "
                f"| mix[{format_type_counts(epoch_type_counts)}] "
                f"| {elapsed:.1f}s"
            )

    if best_epoch > 0:
        best_ckpt = torch.load(run_dir / TEMPORAL_BEST_NAME, map_location=device, weights_only=False)
        temporal_state = best_ckpt["state_dict"]
        base = model.AeroTAF
        if "obs_feature_norm" in temporal_state and getattr(base, "_use_feature_normalization", False):
            base.obs_feature_norm.load_state_dict(temporal_state["obs_feature_norm"])
            base.act_feature_norm.load_state_dict(temporal_state["act_feature_norm"])
        layer = base.AeroTAF
        layer.K_module.load_state_dict(temporal_state["K_module"])
        layer.Q_module.load_state_dict(temporal_state["Q_module"])
        layer.V_module.load_state_dict(temporal_state["V_module"])
        layer.attn_output_module.load_state_dict(temporal_state["attn_output_module"])
        layer.norm.load_state_dict(temporal_state["norm"])
        layer.time_K_module.load_state_dict(temporal_state["time_K_module"])
        layer.time_Q_module.load_state_dict(temporal_state["time_Q_module"])
        layer.time_V_module.load_state_dict(temporal_state["time_V_module"])
        layer.time_attn_output_module.load_state_dict(temporal_state["time_attn_output_module"])
        layer.time_attn_norm.load_state_dict(temporal_state["time_attn_norm"])
        layer.time_ffn_module.load_state_dict(temporal_state["time_ffn_module"])
        layer.time_ffn_norm.load_state_dict(temporal_state["time_ffn_norm"])

        threat_ckpt = torch.load(run_dir / THREAT_BEST_NAME, map_location=device, weights_only=False)
        attack_ckpt = torch.load(run_dir / ATTACK_BEST_NAME, map_location=device, weights_only=False)
        layer.threat_output_module.load_state_dict(threat_ckpt["state_dict"])
        layer.attack_output_module.load_state_dict(attack_ckpt["state_dict"])

    test_natural_metrics = evaluate(model, test_natural_loader, device, args)
    test_key_metrics = evaluate(model, test_key_loader, device, args)
    summary = {
        "best_epoch": int(best_epoch),
        "best_mini": int(best_mini),
        "best_score": float(best_score),
        "test_natural": test_natural_metrics,
        "test_key": test_key_metrics,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logging.info("-" * 72)
    logging.info(
        f"[TEST] natural={test_natural_metrics['loss']:.4f} "
        f"| key={test_key_metrics['loss']:.4f} "
        f"| keyR(t/a)=({test_key_metrics['threat_r']:.3f}/{test_key_metrics['attack_r']:.3f})"
    )
    logging.info(f"[DONE] best_epoch={best_epoch:03d} best_score={best_score:.6f}")
    logging.info(f"summary       : {normalize_path(summary_path)}")


if __name__ == "__main__":
    default_args = [
        "--dataset-dir", "datasets/aerotaf/4v4_shoot_mappo_pool_stage1/processed_stage1_K20_annotated",
        "--experiment-name", "AeroTAF-ATTN-Split-K20",
        "--save-root", "scripts/results/AeroTAF_ATTN",
        "--seed", "1",
        "--n-training-threads", "1",
        "--epochs", "20",
        "--mini-windows", "1000",
        "--lr-temporal", "3e-5",
        "--lr-threat", "1e-4",
        "--lr-attack", "1e-4",
        "--weight-decay", "1e-4",
        "--max-grad-norm", "2.0",
        "--temporal-loss-weight", "0.5",
        "--threat-loss-weight", "1.0",
        "--attack-loss-weight", "8.0",
        "--num-workers", "0",
        "--log-interval", "1",
        "--chunk-length", "50",
        "--key-stride", "5",
        "--background-stride", "50",
        "--train-ratio-event", "0.50",
        "--train-ratio-high-change", "0.20",
        "--train-ratio-high-field", "0.20",
        "--train-ratio-background", "0.10",
        "--val-natural-score-weight", "0.40",
        "--val-key-score-weight", "0.60",
        "--num-agents", "4",
        "--activation-id", "1",
        "--KQ-hidden-size", "128 128",
        "--V-hidden-size", "128 128",
        "--AeroTAF-out-hidden-size", "128",
        "--num-heads", "4",
        "--time-head-num", "4",
        "--use-feature-normalization",
        # "--cuda",
        # "--cuda-device-id", "0",
    ]
    main(sys.argv[1:] if len(sys.argv) > 1 else default_args)
