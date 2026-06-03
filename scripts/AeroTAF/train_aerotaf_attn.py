#!/usr/bin/env python
import argparse
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
    from torch.utils.data import DataLoader, Dataset, Subset
except ModuleNotFoundError as exc:
    logging.info(f"Error: missing dependency: {exc}")
    logging.info("Please activate the same Python environment used by this project, then run this script again.")
    sys.exit(1)

from algorithms.utils.AeroTAF_ATTN import AeroTAFATTNBase
from scripts.AeroTAF.collector.path_utils import normalize_path, resolve_project_path


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
    optimizers = {}

    if backbone_params:
        optimizers["backbone"] = torch.optim.Adam(
            backbone_params,
            lr=args.backbone_lr,
            weight_decay=args.weight_decay,
        )
    if threat_head_params:
        optimizers["threat_head"] = torch.optim.Adam(
            threat_head_params,
            lr=args.threat_head_lr,
            weight_decay=args.weight_decay,
        )
    if attack_head_params:
        optimizers["attack_head"] = torch.optim.Adam(
            attack_head_params,
            lr=args.attack_head_lr,
            weight_decay=args.weight_decay,
        )

    return optimizers, backbone_params, threat_head_params, attack_head_params


class ProcessedSplitStore:
    def __init__(self, npz_path, sample_stride=1, use_temporal_loss=True):
        if sample_stride < 1:
            raise ValueError(f"sample_stride must be >= 1, got {sample_stride}")
        self.use_temporal_loss = bool(use_temporal_loss)

        with np.load(npz_path, allow_pickle=True) as data:
            required_keys = ["obs", "actions", "threat_targets", "attack_targets", "episode_lengths"]
            if self.use_temporal_loss:
                required_keys.append("temporal_targets")
            missing = [key for key in required_keys if key not in data.files]
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
            self.episode_lengths = data["episode_lengths"].astype(np.int32, copy=False)
            self.episode_ids = data["episode_ids"].astype(np.int32, copy=False) if "episode_ids" in data.files else None
            self.random_seeds = data["random_seeds"].astype(np.int32, copy=False) if "random_seeds" in data.files else None
            self.source_files = data["source_files"].astype(object, copy=False) if "source_files" in data.files else None

        total_steps = int(self.episode_lengths.sum())
        if total_steps != self.obs.shape[0]:
            raise ValueError(
                f"{npz_path}: episode_lengths sum {total_steps} does not match obs length {self.obs.shape[0]}"
            )
        if self.obs.shape[0] != self.actions.shape[0]:
            raise ValueError(f"{npz_path}: obs/actions length mismatch")
        if self.obs.shape[0] != self.threat_targets.shape[0]:
            raise ValueError(f"{npz_path}: obs/threat_targets length mismatch")
        if self.obs.shape[0] != self.attack_targets.shape[0]:
            raise ValueError(f"{npz_path}: obs/attack_targets length mismatch")
        if self.use_temporal_loss and self.temporal_targets is None:
            raise KeyError(f"{npz_path}: temporal_targets is required when use_temporal_loss=True")
        if self.temporal_targets is not None and self.obs.shape[0] != self.temporal_targets.shape[0]:
            raise ValueError(f"{npz_path}: obs/temporal_targets length mismatch")

        self.sample_stride = int(sample_stride)
        self.original_step_count = int(self.obs.shape[0])
        self.offsets = []
        self.kept_episode_lengths = []
        start = 0
        kept_steps = 0
        for length in self.episode_lengths.tolist():
            end = start + int(length)
            self.offsets.append((start, end))
            kept_len = len(range(start, end, self.sample_stride))
            self.kept_episode_lengths.append(kept_len)
            kept_steps += kept_len
            start = end
        self.kept_step_count = int(kept_steps)

    def __len__(self):
        return len(self.offsets)

    def episode(self, index):
        start, end = self.offsets[index]
        keep = slice(start, end, self.sample_stride)
        meta = {
            "episode_index": index,
            "episode_id": int(self.episode_ids[index]) if self.episode_ids is not None else index,
            "random_seed": int(self.random_seeds[index]) if self.random_seeds is not None else -1,
            "source_file": str(self.source_files[index]) if self.source_files is not None else "",
        }
        return {
            "obs": self.obs[keep],
            "actions": self.actions[keep],
            "threat_targets": self.threat_targets[keep],
            "attack_targets": self.attack_targets[keep],
            "temporal_targets": self.temporal_targets[keep] if self.temporal_targets is not None else None,
            "length": self.kept_episode_lengths[index],
            "time_offset": 0,
            "meta": meta,
        }


class AeroTAFATTNChunkDataset(Dataset):
    def __init__(self, store, chunk_length, chunk_stride):
        self.store = store
        self.chunk_length = int(chunk_length)
        self.chunk_stride = int(chunk_stride)
        if self.chunk_length < 1:
            raise ValueError(f"chunk_length must be >= 1, got {self.chunk_length}")
        if self.chunk_stride < 1:
            raise ValueError(f"chunk_stride must be >= 1, got {self.chunk_stride}")

        self.chunks = []
        for episode_index in range(len(self.store)):
            episode = self.store.episode(episode_index)
            episode_length = int(episode["length"])

            if episode_length <= self.chunk_length:
                self.chunks.append((episode_index, 0, episode_length))
                continue

            starts = list(range(0, episode_length - self.chunk_length + 1, self.chunk_stride))
            final_start = episode_length - self.chunk_length
            if starts[-1] != final_start:
                starts.append(final_start)

            for start in starts:
                self.chunks.append((episode_index, start, start + self.chunk_length))

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, index):
        episode_index, start, end = self.chunks[index]
        episode = self.store.episode(episode_index)
        temporal_targets = episode["temporal_targets"]
        return (
            episode["obs"][start:end],
            episode["actions"][start:end],
            episode["threat_targets"][start:end],
            episode["attack_targets"][start:end],
            temporal_targets[start:end] if temporal_targets is not None else None,
            end - start,
            start,
        )


def episode_collate_fn(batch):
    if len(batch) != 1:
        raise ValueError("Current ATTN chunk training script supports batch_size=1 only.")
    return batch[0]


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


def build_epoch_minibatch_subsets(dataset, mini_batches, epoch, seed):
    total_chunks = len(dataset)
    if total_chunks == 0:
        raise ValueError("Empty chunk dataset.")

    if mini_batches <= 1 or total_chunks == 1:
        return [dataset]

    effective_batches = min(int(mini_batches), total_chunks)
    rng = random.Random(seed + epoch * 9973)
    indices = list(range(total_chunks))
    rng.shuffle(indices)
    split_indices = np.array_split(np.asarray(indices, dtype=np.int64), effective_batches)
    return [Subset(dataset, split.tolist()) for split in split_indices if len(split) > 0]


def build_loader(dataset, batch_size, shuffle, num_workers, device):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=episode_collate_fn,
        drop_last=False,
    )


def compute_r_value(sse, target_sum, target_square_sum, count):
    if count <= 0:
        return 0.0
    sst = target_square_sum - target_sum * target_sum / count
    if sst <= 1e-12:
        return 0.0
    return 1.0 - sse / sst


def prepare_episode_batch(obs, actions, threat_targets, attack_targets, temporal_targets, seq_len, time_offset, device):
    seq_len = int(seq_len)
    time_offset = int(time_offset)

    obs = torch.as_tensor(obs, dtype=torch.float32, device=device).reshape(seq_len, obs.shape[1], -1)
    actions = torch.as_tensor(actions, dtype=torch.float32, device=device).reshape(seq_len, actions.shape[1], -1)
    threat_targets = torch.as_tensor(threat_targets, dtype=torch.float32, device=device).reshape(seq_len, -1)
    attack_targets = torch.as_tensor(attack_targets, dtype=torch.float32, device=device).reshape(seq_len, -1)
    if temporal_targets is not None:
        temporal_targets = torch.as_tensor(temporal_targets, dtype=torch.float32, device=device).reshape(
            seq_len,
            temporal_targets.shape[1],
            -1,
        )

    obs = obs.reshape(seq_len * obs.shape[1], -1)
    actions = actions.reshape(seq_len * actions.shape[1], -1)
    if temporal_targets is not None:
        temporal_targets = temporal_targets.reshape(seq_len * temporal_targets.shape[1], -1)
    return obs, actions, threat_targets, attack_targets, temporal_targets, seq_len, time_offset


def evaluate(
    model,
    data_loader,
    device,
    threat_loss_weight,
    attack_loss_weight,
    temporal_loss_weight,
    use_temporal_loss,
    return_stats=False,
):
    model.eval()
    total_loss = 0.0
    total_threat_loss = 0.0
    total_attack_loss = 0.0
    total_temporal_loss = 0.0
    total_raw_threat_loss = 0.0
    total_raw_attack_loss = 0.0
    total_raw_temporal_loss = 0.0
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
        for obs, actions, threat_targets, attack_targets, temporal_targets, seq_len, time_offset in data_loader:
            obs, actions, threat_targets, attack_targets, temporal_targets, seq_len, time_offset = prepare_episode_batch(
                obs,
                actions,
                threat_targets,
                attack_targets,
                temporal_targets,
                seq_len,
                time_offset,
                device,
            )
            temporal_pred, threat_pred, attack_pred = model(obs, actions, seq_len=seq_len, time_offset=time_offset)

            threat_loss = F.mse_loss(threat_pred, threat_targets)
            attack_loss = F.mse_loss(attack_pred, attack_targets)
            if use_temporal_loss:
                temporal_loss = F.mse_loss(temporal_pred, temporal_targets)
            else:
                temporal_loss = torch.zeros((), dtype=torch.float32, device=device)
            loss = (
                threat_loss_weight * threat_loss
                + attack_loss_weight * attack_loss
                + temporal_loss_weight * temporal_loss
            )

            threat_error = threat_pred - threat_targets
            attack_error = attack_pred - attack_targets
            threat_sse += torch.sum(threat_error * threat_error).item()
            attack_sse += torch.sum(attack_error * attack_error).item()
            threat_sum += torch.sum(threat_targets).item()
            attack_sum += torch.sum(attack_targets).item()
            threat_square_sum += torch.sum(threat_targets * threat_targets).item()
            attack_square_sum += torch.sum(attack_targets * attack_targets).item()
            threat_count += threat_targets.numel()
            attack_count += attack_targets.numel()

            total_loss += loss.item() * seq_len
            total_threat_loss += threat_loss_weight * threat_loss.item() * seq_len
            total_attack_loss += attack_loss_weight * attack_loss.item() * seq_len
            total_temporal_loss += temporal_loss_weight * temporal_loss.item() * seq_len
            total_raw_threat_loss += threat_loss.item() * seq_len
            total_raw_attack_loss += attack_loss.item() * seq_len
            total_raw_temporal_loss += temporal_loss.item() * seq_len
            total_steps += seq_len

    metrics = {
        "loss": total_loss / max(total_steps, 1),
        "threat_loss": total_threat_loss / max(total_steps, 1),
        "attack_loss": total_attack_loss / max(total_steps, 1),
        "temporal_loss": total_temporal_loss / max(total_steps, 1),
        "raw_threat_loss": total_raw_threat_loss / max(total_steps, 1),
        "raw_attack_loss": total_raw_attack_loss / max(total_steps, 1),
        "raw_temporal_loss": total_raw_temporal_loss / max(total_steps, 1),
        "valid_steps": int(total_steps),
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


def merge_eval_metrics(metrics_list):
    total_samples = sum(item["samples"] for item in metrics_list)
    if total_samples <= 0:
        return {
            "loss": 0.0,
            "threat_loss": 0.0,
            "attack_loss": 0.0,
            "temporal_loss": 0.0,
            "raw_threat_loss": 0.0,
            "raw_attack_loss": 0.0,
            "raw_temporal_loss": 0.0,
            "valid_steps": 0,
            "threat_r": 0.0,
            "attack_r": 0.0,
            "samples": 0,
            "mini_batch_count": 0,
        }

    threat_sse = sum(item["threat_sse"] for item in metrics_list)
    attack_sse = sum(item["attack_sse"] for item in metrics_list)
    threat_sum = sum(item["threat_sum"] for item in metrics_list)
    attack_sum = sum(item["attack_sum"] for item in metrics_list)
    threat_square_sum = sum(item["threat_square_sum"] for item in metrics_list)
    attack_square_sum = sum(item["attack_square_sum"] for item in metrics_list)
    threat_count = sum(item["threat_count"] for item in metrics_list)
    attack_count = sum(item["attack_count"] for item in metrics_list)

    merged = {
        "loss": sum(item["loss"] * item["samples"] for item in metrics_list) / total_samples,
        "threat_loss": sum(item["threat_loss"] * item["samples"] for item in metrics_list) / total_samples,
        "attack_loss": sum(item["attack_loss"] * item["samples"] for item in metrics_list) / total_samples,
        "temporal_loss": sum(item["temporal_loss"] * item["samples"] for item in metrics_list) / total_samples,
        "raw_threat_loss": sum(item["raw_threat_loss"] * item["samples"] for item in metrics_list) / total_samples,
        "raw_attack_loss": sum(item["raw_attack_loss"] * item["samples"] for item in metrics_list) / total_samples,
        "raw_temporal_loss": sum(item["raw_temporal_loss"] * item["samples"] for item in metrics_list) / total_samples,
        "valid_steps": int(total_samples),
        "threat_r": compute_r_value(threat_sse, threat_sum, threat_square_sum, threat_count),
        "attack_r": compute_r_value(attack_sse, attack_sum, attack_square_sum, attack_count),
        "samples": int(total_samples),
        "mini_batch_count": len(metrics_list),
    }
    return merged


def train_one_round(
    model,
    optimizer,
    optimizers,
    train_mode,
    data_loader,
    device,
    max_grad_norm,
    threat_loss_weight,
    attack_loss_weight,
    temporal_loss_weight,
    use_temporal_loss,
    backbone_params=None,
    threat_head_params=None,
    attack_head_params=None,
):
    model.train()
    total_loss = 0.0
    total_threat_loss = 0.0
    total_attack_loss = 0.0
    total_temporal_loss = 0.0
    total_raw_threat_loss = 0.0
    total_raw_attack_loss = 0.0
    total_raw_temporal_loss = 0.0
    total_steps = 0

    for obs, actions, threat_targets, attack_targets, temporal_targets, seq_len, time_offset in data_loader:
        obs, actions, threat_targets, attack_targets, temporal_targets, seq_len, time_offset = prepare_episode_batch(
            obs,
            actions,
            threat_targets,
            attack_targets,
            temporal_targets,
            seq_len,
            time_offset,
            device,
        )
        temporal_pred, threat_pred, attack_pred = model(obs, actions, seq_len=seq_len, time_offset=time_offset)

        threat_loss = F.mse_loss(threat_pred, threat_targets)
        attack_loss = F.mse_loss(attack_pred, attack_targets)
        if use_temporal_loss:
            temporal_loss = F.mse_loss(temporal_pred, temporal_targets)
        else:
            temporal_loss = torch.zeros((), dtype=torch.float32, device=device)
        loss = (
            threat_loss_weight * threat_loss
            + attack_loss_weight * attack_loss
            + temporal_loss_weight * temporal_loss
        )

        if train_mode in ("joint", "heads_only"):
            optimizer.zero_grad()
            loss.backward()
            if max_grad_norm is not None and max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
        else:
            for opt in optimizers.values():
                opt.zero_grad()
            zero_param_grads(backbone_params)
            zero_param_grads(threat_head_params)
            zero_param_grads(attack_head_params)

            need_backbone = use_temporal_loss and temporal_loss_weight > 0.0 and len(backbone_params) > 0
            need_threat = threat_loss_weight > 0.0 and len(threat_head_params) > 0
            need_attack = attack_loss_weight > 0.0 and len(attack_head_params) > 0

            if need_backbone:
                retain_graph = need_threat or need_attack
                backbone_grads = torch.autograd.grad(
                    temporal_loss_weight * temporal_loss,
                    backbone_params,
                    retain_graph=retain_graph,
                    allow_unused=False,
                )
                assign_grads(backbone_params, backbone_grads)

            if need_threat:
                retain_graph = need_attack
                threat_grads = torch.autograd.grad(
                    threat_loss_weight * threat_loss,
                    threat_head_params,
                    retain_graph=retain_graph,
                    allow_unused=False,
                )
                assign_grads(threat_head_params, threat_grads)

            if need_attack:
                attack_grads = torch.autograd.grad(
                    attack_loss_weight * attack_loss,
                    attack_head_params,
                    retain_graph=False,
                    allow_unused=False,
                )
                assign_grads(attack_head_params, attack_grads)

            if need_backbone:
                clip_param_grads(backbone_params, max_grad_norm)
                optimizers["backbone"].step()
            if need_threat:
                clip_param_grads(threat_head_params, max_grad_norm)
                optimizers["threat_head"].step()
            if need_attack:
                clip_param_grads(attack_head_params, max_grad_norm)
                optimizers["attack_head"].step()

        total_loss += loss.item() * seq_len
        total_threat_loss += threat_loss_weight * threat_loss.item() * seq_len
        total_attack_loss += attack_loss_weight * attack_loss.item() * seq_len
        total_temporal_loss += temporal_loss_weight * temporal_loss.item() * seq_len
        total_raw_threat_loss += threat_loss.item() * seq_len
        total_raw_attack_loss += attack_loss.item() * seq_len
        total_raw_temporal_loss += temporal_loss.item() * seq_len
        total_steps += seq_len

    return {
        "loss": total_loss / max(total_steps, 1),
        "threat_loss": total_threat_loss / max(total_steps, 1),
        "attack_loss": total_attack_loss / max(total_steps, 1),
        "temporal_loss": total_temporal_loss / max(total_steps, 1),
        "raw_threat_loss": total_raw_threat_loss / max(total_steps, 1),
        "raw_attack_loss": total_raw_attack_loss / max(total_steps, 1),
        "raw_temporal_loss": total_raw_temporal_loss / max(total_steps, 1),
        "valid_steps": int(total_steps),
        "samples": int(total_steps),
    }


def append_csv_row(path, row, write_header=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(row.keys())
    line = ",".join(str(row[key]) for key in keys)
    if write_header and not path.exists():
        with open(path, "w", encoding="utf-8") as f:
            f.write(",".join(keys) + "\n")
            f.write(line + "\n")
        return
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def save_checkpoint(path, model, optimizer_state, epoch, best_val_loss, args):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer_state,
            "epoch": epoch,
            "best_val_loss": best_val_loss,
            "args": vars(args),
        },
        path,
    )


def build_run_dir(args):
    save_root = resolve_project_path(args.save_root)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_name = f"run-{timestamp}-seed{args.seed}"
    return save_root / args.experiment_name / run_name


def format_round_log(
    epoch,
    epochs,
    mini_batch_index,
    mini_batch_count,
    global_round,
    total_rounds,
    val_metrics,
    train_metrics,
    elapsed,
    train_mode,
):
    return (
        f"\n"
        f"[Epoch {epoch:03d}/{epochs:03d} | Round {global_round:04d}/{total_rounds:04d} | "
        f"Mini-batch {mini_batch_index:02d}/{mini_batch_count:02d} | Mode {train_mode}]\n"
        f"  Val-before-train : loss={val_metrics['loss']:.6f} | threat={val_metrics['threat_loss']:.6f} | "
        f"attack={val_metrics['attack_loss']:.6f} | temporal={val_metrics['temporal_loss']:.6f} | "
        f"threat_R={val_metrics['threat_r']:.4f} | attack_R={val_metrics['attack_r']:.4f} | "
        f"samples={val_metrics['samples']}\n"
        f"  Train            : loss={train_metrics['loss']:.6f} | threat={train_metrics['threat_loss']:.6f} | "
        f"attack={train_metrics['attack_loss']:.6f} | temporal={train_metrics['temporal_loss']:.6f} | "
        f"samples={train_metrics['samples']}\n"
        f"  Time             : {elapsed:.2f}s"
    )


def collect_optimizer_state(train_mode, optimizer, optimizers):
    if train_mode == "split":
        return {name: opt.state_dict() for name, opt in optimizers.items()}
    return optimizer.state_dict()


def dump_config(path, args, dataset_paths, stores, datasets=None):
    payload = {
        "args": vars(args),
        "dataset_paths": {key: normalize_path(value) for key, value in dataset_paths.items()},
        "dataset_summary": {
            "train_steps_original": int(stores["train"].original_step_count),
            "train_steps_kept": int(stores["train"].kept_step_count),
            "val_steps_original": int(stores["val"].original_step_count),
            "val_steps_kept": int(stores["val"].kept_step_count),
            "test_steps_original": int(stores["test"].original_step_count),
            "test_steps_kept": int(stores["test"].kept_step_count),
            "train_episodes": int(len(stores["train"])),
            "val_episodes": int(len(stores["val"])),
            "test_episodes": int(len(stores["test"])),
        },
    }
    if datasets is not None:
        payload["chunk_summary"] = {
            "chunk_length": int(args.chunk_length),
            "chunk_stride": int(args.chunk_stride),
            "mini_batches": int(args.mini_batches),
            "train_chunks": int(len(datasets["train"])),
            "val_chunks": int(len(datasets["val"])),
            "test_chunks": int(len(datasets["test"])),
        }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def get_parser():
    parser = argparse.ArgumentParser(description="Train AeroTAF_ATTN on processed stage-1 datasets with sliding-window chunks.")
    parser.add_argument("--dataset-dir", type=str, required=True, help="Processed dataset directory.")
    parser.add_argument("--train-file", type=str, default="train.npz", help="Training split filename.")
    parser.add_argument("--val-file", type=str, default="val.npz", help="Validation split filename.")
    parser.add_argument("--test-file", type=str, default="test.npz", help="Test split filename.")
    parser.add_argument("--experiment-name", type=str, default="AeroTAF-ATTN-Stage1-K20-SlidingWindow", help="Experiment name.")
    parser.add_argument("--save-root", type=str, default="scripts/results/AeroTAF_ATTN", help="Checkpoint root directory.")
    parser.add_argument("--seed", type=int, default=1, help="Random seed.")
    parser.add_argument("--cuda", action="store_true", default=False, help="Use CUDA if available.")
    parser.add_argument("--cuda-device-id", type=int, default=0, help="CUDA device id.")
    parser.add_argument("--n-training-threads", type=int, default=1, help="Torch training thread count.")
    parser.add_argument("--batch-size", type=int, default=1, help="Chunk batch size. Current script supports 1 only.")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs.")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Adam weight decay.")
    parser.add_argument("--max-grad-norm", type=float, default=2.0, help="Gradient clipping max norm.")
    parser.add_argument(
        "--train-mode",
        type=str,
        default="joint",
        choices=("heads_only", "joint", "split"),
        help="Training mode: heads_only uses threat/attack only; joint trains temporal+threat+attack together; split updates backbone by temporal loss and heads by their own losses.",
    )
    parser.add_argument("--threat-loss-weight", type=float, default=1.0, help="Weight for threat regression loss.")
    parser.add_argument("--attack-loss-weight", type=float, default=1.0, help="Weight for attack regression loss.")
    parser.add_argument("--temporal-loss-weight", type=float, default=0.05, help="Weight for temporal feature regression loss.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker count.")
    parser.add_argument("--save-interval", type=int, default=10, help="Latest checkpoint save interval.")
    parser.add_argument("--log-interval", type=int, default=1, help="Epoch logging interval.")
    parser.add_argument("--chunk-length", type=int, default=32, help="Sliding-window chunk length on the time axis.")
    parser.add_argument("--chunk-stride", type=int, default=8, help="Sliding-window stride on the time axis.")
    parser.add_argument(
        "--mini-batches",
        type=int,
        default=1,
        help="Split train/val chunks into this many partitions and run one val-train pair per partition in each epoch.",
    )
    parser.add_argument(
        "--sample-stride",
        type=int,
        default=1,
        help="Keep only timesteps whose within-episode index is a multiple of this stride. 1 keeps all steps.",
    )
    parser.add_argument("--num-agents", type=int, default=4, help="Number of ego agents.")
    parser.add_argument("--activation-id", type=int, default=1, help="0:Tanh, 1:ReLU, 2:LeakyReLU, 3:ELU.")
    parser.add_argument("--use-feature-normalization", action="store_true", default=False, help="Use LayerNorm on obs/action inputs.")
    parser.add_argument("--KQ-hidden-size", type=str, default="64 64", help="Spatial/time K/Q MLP hidden sizes.")
    parser.add_argument("--V-hidden-size", type=str, default="64 64", help="Spatial/time V MLP hidden sizes.")
    parser.add_argument("--AeroTAF-out-hidden-size", type=str, default="64", help="Threat/attack head hidden sizes.")
    parser.add_argument("--num-heads", type=int, default=4, help="Spatial attention head count.")
    parser.add_argument("--time-head-num", type=int, default=4, help="Temporal attention head count.")
    return parser


def main(args):
    parser = get_parser()
    all_args = parser.parse_args(args)
    train_mode = all_args.train_mode
    use_temporal_loss = train_mode != "heads_only"

    if all_args.batch_size != 1:
        raise ValueError("Current sliding-window ATTN training supports batch_size=1 only.")

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

    set_seed(all_args.seed)
    torch.set_num_threads(all_args.n_training_threads)
    device = build_device(all_args)

    stores = {
        "train": ProcessedSplitStore(
            dataset_paths["train"],
            sample_stride=all_args.sample_stride,
            use_temporal_loss=use_temporal_loss,
        ),
        "val": ProcessedSplitStore(
            dataset_paths["val"],
            sample_stride=all_args.sample_stride,
            use_temporal_loss=use_temporal_loss,
        ),
        "test": ProcessedSplitStore(
            dataset_paths["test"],
            sample_stride=all_args.sample_stride,
            use_temporal_loss=use_temporal_loss,
        ),
    }

    obs_space, act_space = build_spaces(stores["train"])
    model_args = AeroTAFATTNArgs(all_args)
    model = PPOAeroTAFATTN(model_args, obs_space, act_space, device=device)

    if train_mode == "split":
        optimizers, backbone_params, threat_head_params, attack_head_params = build_split_optimizers(model, all_args)
        optimizer = None
    else:
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=all_args.lr,
            weight_decay=all_args.weight_decay,
        )
        optimizers = {}
        backbone_params = threat_head_params = attack_head_params = None

    datasets = {
        "train": AeroTAFATTNChunkDataset(stores["train"], all_args.chunk_length, all_args.chunk_stride),
        "val": AeroTAFATTNChunkDataset(stores["val"], all_args.chunk_length, all_args.chunk_stride),
        "test": AeroTAFATTNChunkDataset(stores["test"], all_args.chunk_length, all_args.chunk_stride),
    }

    test_loader = build_loader(
        datasets["test"],
        batch_size=all_args.batch_size,
        shuffle=False,
        num_workers=all_args.num_workers,
        device=device,
    )

    run_dir = build_run_dir(all_args)
    run_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = run_dir / "AeroTAF_ATTN_best.pt"
    latest_ckpt_path = run_dir / "AeroTAF_ATTN_latest.pt"
    log_path = run_dir / "train_log.csv"
    config_path = run_dir / "config.json"
    test_metrics_path = run_dir / "test_metrics.json"
    dump_config(config_path, all_args, dataset_paths, stores, datasets)

    effective_mini_batches = max(1, min(all_args.mini_batches, len(datasets["train"]), len(datasets["val"])))
    total_rounds = all_args.epochs * effective_mini_batches

    logging.info(f"device: {device}")
    logging.info(f"dataset dir: {normalize_path(dataset_dir)}")
    logging.info(f"run dir: {normalize_path(run_dir)}")
    logging.info(
        f"sample stride: {all_args.sample_stride} | "
        f"train kept/original={stores['train'].kept_step_count}/{stores['train'].original_step_count} | "
        f"val kept/original={stores['val'].kept_step_count}/{stores['val'].original_step_count} | "
        f"test kept/original={stores['test'].kept_step_count}/{stores['test'].original_step_count}"
    )
    logging.info(
        f"chunk setting: length={all_args.chunk_length}, stride={all_args.chunk_stride} "
        f"(sliding window with overlap={max(all_args.chunk_length - all_args.chunk_stride, 0)})"
    )
    logging.info(
        f"train mode: {train_mode} | use_temporal_loss={int(use_temporal_loss)} | "
        f"mini-batches per epoch={effective_mini_batches} | total paired rounds={total_rounds}"
    )
    logging.info(
        f"loss weights: threat={all_args.threat_loss_weight}, attack={all_args.attack_loss_weight}, "
        f"temporal={all_args.temporal_loss_weight}"
    )
    if train_mode == "split":
        logging.info(
            f"parameter groups: backbone={len(backbone_params)}, threat_head={len(threat_head_params)}, "
            f"attack_head={len(attack_head_params)}"
        )
    logging.info(
        f"train episodes: {len(stores['train'])}, steps: {stores['train'].kept_step_count}, chunks: {len(datasets['train'])}"
    )
    logging.info(
        f"val episodes: {len(stores['val'])}, steps: {stores['val'].kept_step_count}, chunks: {len(datasets['val'])}"
    )
    logging.info(
        f"test episodes: {len(stores['test'])}, steps: {stores['test'].kept_step_count}, chunks: {len(datasets['test'])}"
    )
    logging.info("round order: validate one val subset -> train one train subset")

    best_val_loss = float("inf")

    for epoch in range(1, all_args.epochs + 1):
        train_subsets = build_epoch_minibatch_subsets(datasets["train"], effective_mini_batches, epoch, all_args.seed)
        val_subsets = build_epoch_minibatch_subsets(datasets["val"], effective_mini_batches, epoch, all_args.seed + 100000)

        for mini_batch_index, (val_subset, train_subset) in enumerate(zip(val_subsets, train_subsets), start=1):
            global_round = (epoch - 1) * effective_mini_batches + mini_batch_index
            round_start = time.time()

            val_loader = build_loader(
                val_subset,
                batch_size=all_args.batch_size,
                shuffle=False,
                num_workers=all_args.num_workers,
                device=device,
            )
            val_metrics = evaluate(
                model=model,
                data_loader=val_loader,
                device=device,
                threat_loss_weight=all_args.threat_loss_weight,
                attack_loss_weight=all_args.attack_loss_weight,
                temporal_loss_weight=all_args.temporal_loss_weight,
                use_temporal_loss=use_temporal_loss,
                return_stats=True,
            )

            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                save_checkpoint(
                    best_ckpt_path,
                    model,
                    collect_optimizer_state(train_mode, optimizer, optimizers),
                    global_round - 1,
                    best_val_loss,
                    all_args,
                )
                logging.info(f"best checkpoint updated at round {global_round}: {normalize_path(best_ckpt_path)}")

            train_loader = build_loader(
                train_subset,
                batch_size=all_args.batch_size,
                shuffle=True,
                num_workers=all_args.num_workers,
                device=device,
            )
            train_metrics = train_one_round(
                model=model,
                optimizer=optimizer,
                optimizers=optimizers,
                train_mode=train_mode,
                data_loader=train_loader,
                device=device,
                max_grad_norm=all_args.max_grad_norm,
                threat_loss_weight=all_args.threat_loss_weight,
                attack_loss_weight=all_args.attack_loss_weight,
                temporal_loss_weight=all_args.temporal_loss_weight,
                use_temporal_loss=use_temporal_loss,
                backbone_params=backbone_params,
                threat_head_params=threat_head_params,
                attack_head_params=attack_head_params,
            )
            elapsed = time.time() - round_start

            row = {
                "epoch": epoch,
                "mini_batch_index": mini_batch_index,
                "mini_batch_total": effective_mini_batches,
                "global_round": global_round,
                "total_rounds": total_rounds,
                "train_mode": train_mode,
                "val_before_train_loss": f"{val_metrics['loss']:.8f}",
                "val_before_train_threat_loss": f"{val_metrics['threat_loss']:.8f}",
                "val_before_train_attack_loss": f"{val_metrics['attack_loss']:.8f}",
                "val_before_train_temporal_loss": f"{val_metrics['temporal_loss']:.8f}",
                "val_before_train_raw_threat_loss": f"{val_metrics['raw_threat_loss']:.8f}",
                "val_before_train_raw_attack_loss": f"{val_metrics['raw_attack_loss']:.8f}",
                "val_before_train_raw_temporal_loss": f"{val_metrics['raw_temporal_loss']:.8f}",
                "val_before_train_threat_r": f"{val_metrics['threat_r']:.8f}",
                "val_before_train_attack_r": f"{val_metrics['attack_r']:.8f}",
                "val_samples": val_metrics["samples"],
                "train_loss": f"{train_metrics['loss']:.8f}",
                "train_threat_loss": f"{train_metrics['threat_loss']:.8f}",
                "train_attack_loss": f"{train_metrics['attack_loss']:.8f}",
                "train_temporal_loss": f"{train_metrics['temporal_loss']:.8f}",
                "train_raw_threat_loss": f"{train_metrics['raw_threat_loss']:.8f}",
                "train_raw_attack_loss": f"{train_metrics['raw_attack_loss']:.8f}",
                "train_raw_temporal_loss": f"{train_metrics['raw_temporal_loss']:.8f}",
                "train_samples": train_metrics["samples"],
                "time_sec": f"{elapsed:.4f}",
            }
            append_csv_row(log_path, row, write_header=(global_round == 1))

            if (
                epoch % all_args.log_interval == 0
                or epoch == 1
                or epoch == all_args.epochs
                or mini_batch_index == 1
            ):
                logging.info(
                    format_round_log(
                        epoch=epoch,
                        epochs=all_args.epochs,
                        mini_batch_index=mini_batch_index,
                        mini_batch_count=effective_mini_batches,
                        global_round=global_round,
                        total_rounds=total_rounds,
                        val_metrics=val_metrics,
                        train_metrics=train_metrics,
                        elapsed=elapsed,
                        train_mode=train_mode,
                    )
                )

            if global_round % all_args.save_interval == 0 or global_round == total_rounds:
                save_checkpoint(
                    latest_ckpt_path,
                    model,
                    collect_optimizer_state(train_mode, optimizer, optimizers),
                    global_round,
                    best_val_loss,
                    all_args,
                )
                logging.info(f"latest checkpoint saved: {normalize_path(latest_ckpt_path)}")

    final_val_loader = build_loader(
        datasets["val"],
        batch_size=all_args.batch_size,
        shuffle=False,
        num_workers=all_args.num_workers,
        device=device,
    )
    final_val_metrics = evaluate(
        model=model,
        data_loader=final_val_loader,
        device=device,
        threat_loss_weight=all_args.threat_loss_weight,
        attack_loss_weight=all_args.attack_loss_weight,
        temporal_loss_weight=all_args.temporal_loss_weight,
        use_temporal_loss=use_temporal_loss,
        return_stats=True,
    )
    if final_val_metrics["loss"] < best_val_loss:
        best_val_loss = final_val_metrics["loss"]
        save_checkpoint(
            best_ckpt_path,
            model,
            collect_optimizer_state(train_mode, optimizer, optimizers),
            total_rounds,
            best_val_loss,
            all_args,
        )
        logging.info(f"best checkpoint updated after final val: {normalize_path(best_ckpt_path)}")

    logging.info(
        f"final val loss={final_val_metrics['loss']:.6f} | threat={final_val_metrics['threat_loss']:.6f} | "
        f"attack={final_val_metrics['attack_loss']:.6f} | temporal={final_val_metrics['temporal_loss']:.6f} | "
        f"threat_R={final_val_metrics['threat_r']:.4f} | attack_R={final_val_metrics['attack_r']:.4f}"
    )

    if best_ckpt_path.exists():
        checkpoint = torch.load(best_ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        logging.info(f"loaded best checkpoint for final test: {normalize_path(best_ckpt_path)}")

    test_metrics = evaluate(
        model=model,
        data_loader=test_loader,
        device=device,
        threat_loss_weight=all_args.threat_loss_weight,
        attack_loss_weight=all_args.attack_loss_weight,
        temporal_loss_weight=all_args.temporal_loss_weight,
        use_temporal_loss=use_temporal_loss,
        return_stats=True,
    )
    with open(test_metrics_path, "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2, ensure_ascii=False)

    logging.info(
        f"test loss={test_metrics['loss']:.6f} | threat={test_metrics['threat_loss']:.6f} | "
        f"attack={test_metrics['attack_loss']:.6f} | temporal={test_metrics['temporal_loss']:.6f} | "
        f"threat_R={test_metrics['threat_r']:.4f} | attack_R={test_metrics['attack_r']:.4f}"
    )
    logging.info(f"test metrics saved: {normalize_path(test_metrics_path)}")
    logging.info("Done.")


if __name__ == "__main__":
    default_args = [
        "--dataset-dir", "datasets/aerotaf/4v4_shoot_mappo_pool_stage1/processed_stage1_K20_field_temporal",
        "--experiment-name", "AeroTAF-ATTN-Stage1-K20-SlidingWindow",
        "--save-root", "scripts/results/AeroTAF_ATTN",
        "--seed", "1",
        "--n-training-threads", "1",
        "--batch-size", "1",
        "--epochs", "10",
        "--mini-batches", "10",
        "--lr", "3e-5",
        "--weight-decay", "1e-4",
        "--max-grad-norm", "2.0",
        "--train-mode", "joint",
        "--threat-loss-weight", "1.0",
        "--attack-loss-weight", "8.0",
        "--temporal-loss-weight", "0.5",
        "--num-workers", "0",
        "--save-interval", "10",
        "--log-interval", "1",
        "--chunk-length", "50",
        "--chunk-stride", "30",
        "--sample-stride", "1",
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
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main(sys.argv[1:] if len(sys.argv) > 1 else default_args)
