#!/usr/bin/env python
import argparse
import csv
import json
import logging
import random
import sys
import time
from collections import OrderedDict
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
from scripts.AeroTAF.data.schema import CATEGORY_NAMES, CATEGORY_STABLE


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

    def forward(self, obs, actions, seq_len, time_offset=0):
        obs = torch.as_tensor(obs, **self.tpdv)
        actions = torch.as_tensor(actions, **self.tpdv)
        _, threat_output, attack_output = self.AeroTAF(
            obs,
            actions,
            seq_len=seq_len,
            time_offset=time_offset,
        )
        return threat_output, attack_output


class RawEpisodeCache:
    def __init__(self, max_items=4):
        self.max_items = max(1, int(max_items))
        self.cache = OrderedDict()

    def get(self, raw_path):
        key = str(raw_path)
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        with np.load(raw_path, allow_pickle=True) as data:
            missing = [key for key in ("obs", "actions") if key not in data.files]
            if missing:
                raise KeyError(f"{raw_path} missing keys: {missing}")
            obs = data["obs"].astype(np.float32, copy=False)
            actions = data["actions"].astype(np.float32, copy=False)
        self.cache[key] = (obs, actions)
        if len(self.cache) > self.max_items:
            self.cache.popitem(last=False)
        return self.cache[key]


def object_array_to_strings(values):
    return np.asarray([str(v) for v in np.asarray(values, dtype=object).reshape(-1)], dtype=object)


def np_scalar_to_string(value):
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return str(value.item())
        if value.size == 1:
            return str(value.reshape(-1)[0].item())
    return str(value)


class AeroTAFATTNWindowDataset(Dataset):
    def __init__(
        self,
        npz_path,
        history_windows,
        raw_cache_size=4,
        parent_indices=None,
        name=None,
        shared_target=None,
    ):
        self.path = str(npz_path)
        self.name = str(name) if name is not None else Path(npz_path).stem
        self.history_windows = max(1, int(history_windows))
        self.raw_cache_size = int(raw_cache_size)
        self.raw_cache = RawEpisodeCache(self.raw_cache_size)

        if shared_target is None:
            with np.load(npz_path, allow_pickle=True) as split_data:
                if "all_target_indices" not in split_data.files:
                    raise KeyError(f"{npz_path} missing all_target_indices. Rebuild with build_targets_detail.py")
                split_indices = split_data["all_target_indices"].astype(np.int64, copy=False).reshape(-1)
                all_target_file = (
                    np_scalar_to_string(split_data["all_target_file"])
                    if "all_target_file" in split_data.files
                    else "all_target.npz"
                )
            all_target_path = Path(npz_path).parent / all_target_file
            if not all_target_path.exists():
                all_target_path = resolve_project_path(all_target_file)
            self._load_all_target(all_target_path)
            self.all_parent_indices = split_indices
        else:
            self._copy_shared_target(shared_target)
            self.all_parent_indices = np.asarray(parent_indices, dtype=np.int64).reshape(-1)

        if parent_indices is not None and shared_target is None:
            self.all_parent_indices = np.asarray(parent_indices, dtype=np.int64).reshape(-1)

        self._validate_parent_indices(self.all_parent_indices)
        self.all_categories = self.sample_category[self.all_parent_indices]
        self.active_parent_indices = self.all_parent_indices.copy()
        self.categories = self.all_categories.copy()
        self._infer_shapes()
        self.num_windows = int(len(self.active_parent_indices))
        self.window_length = self.history_windows
        self.num_steps = int(len(self.active_parent_indices))

    def _load_all_target(self, all_target_path):
        self.all_target_path = Path(all_target_path)
        if not self.all_target_path.exists():
            raise FileNotFoundError(f"all_target file not found: {self.all_target_path}")
        with np.load(self.all_target_path, allow_pickle=True) as data:
            required = ["source_files", "raw_file_indices", "time_indices", "sample_category", "threat_targets", "attack_targets"]
            missing = [key for key in required if key not in data.files]
            if missing:
                raise KeyError(f"{self.all_target_path} missing keys: {missing}")
            self.source_files = object_array_to_strings(data["source_files"])
            self.raw_file_indices = data["raw_file_indices"].astype(np.int64, copy=False).reshape(-1)
            self.time_indices = data["time_indices"].astype(np.int64, copy=False).reshape(-1)
            self.sample_category = data["sample_category"].astype(np.int64, copy=False).reshape(-1)
            self.threat_targets = data["threat_targets"].astype(np.float32, copy=False).reshape(-1, 1)
            self.attack_targets = data["attack_targets"].astype(np.float32, copy=False).reshape(-1, 1)
            self.category_names = (
                [str(x) for x in data["sample_category_names"].tolist()]
                if "sample_category_names" in data.files
                else list(CATEGORY_NAMES)
            )

    def _copy_shared_target(self, other):
        self.all_target_path = other.all_target_path
        self.source_files = other.source_files
        self.raw_file_indices = other.raw_file_indices
        self.time_indices = other.time_indices
        self.sample_category = other.sample_category
        self.threat_targets = other.threat_targets
        self.attack_targets = other.attack_targets
        self.category_names = other.category_names

    def _validate_parent_indices(self, indices):
        indices = np.asarray(indices, dtype=np.int64).reshape(-1)
        parent_size = int(self.time_indices.shape[0])
        if np.any(indices < 0) or np.any(indices >= parent_size):
            bad = indices[(indices < 0) | (indices >= parent_size)][:10].tolist()
            raise ValueError(f"{self.name}: all_target_indices out of range, examples={bad}, parent_size={parent_size}")

    def _resolve_raw_path(self, row):
        raw_file_index = int(self.raw_file_indices[int(row)])
        if raw_file_index < 0 or raw_file_index >= self.source_files.shape[0]:
            raise IndexError(f"bad raw_file_index={raw_file_index} for all_target row={row}")
        return resolve_project_path(self.source_files[raw_file_index])

    def _infer_shapes(self):
        if len(self.all_parent_indices) <= 0:
            raise ValueError(f"{self.name}: empty split")
        row = int(self.all_parent_indices[0])
        obs, actions = self.raw_cache.get(self._resolve_raw_path(row))
        t = int(self.time_indices[row])
        if obs.ndim != 3 or actions.ndim != 3:
            raise ValueError(f"raw obs/actions must be [T, agents, dim], got {obs.shape}, {actions.shape}")
        self.num_agents = int(obs.shape[1])
        self.obs_dim = int(obs.shape[-1])
        self.act_dim = int(actions.shape[-1])
        if t < 0 or t >= obs.shape[0] or t >= actions.shape[0]:
            raise IndexError(f"bad time_index={t} for first sample")

    def __len__(self):
        return int(self.active_parent_indices.shape[0])

    def __getitem__(self, index):
        row = int(self.active_parent_indices[int(index)])
        obs, actions = self.raw_cache.get(self._resolve_raw_path(row))
        t = int(self.time_indices[row])
        if t < 0 or t >= obs.shape[0] or t >= actions.shape[0]:
            raise IndexError(f"bad time_index={t} for row={row}")
        start = max(0, t - self.history_windows + 1)
        return (
            obs[start : t + 1],
            actions[start : t + 1],
            self.threat_targets[row],
            self.attack_targets[row],
            np.asarray(start, dtype=np.int64),
        )

    def set_epoch_sample(self, stable_sample_ratio, seed, epoch, shuffle=True):
        rng = np.random.default_rng(int(seed) + int(epoch) * 10007)
        key_positions = np.flatnonzero(self.all_categories != CATEGORY_STABLE)
        stable_positions = np.flatnonzero(self.all_categories == CATEGORY_STABLE)
        ratio = float(stable_sample_ratio)
        if ratio >= 1.0:
            stable_take = stable_positions
        elif ratio <= 0.0 or stable_positions.size == 0:
            stable_take = np.asarray([], dtype=np.int64)
        else:
            stable_count = int(np.ceil(stable_positions.size * ratio))
            stable_count = max(1, min(stable_count, stable_positions.size))
            stable_take = rng.choice(stable_positions, size=stable_count, replace=False).astype(np.int64)
        positions = np.concatenate((key_positions, stable_take), axis=0).astype(np.int64, copy=False)
        if shuffle and positions.size > 0:
            rng.shuffle(positions)
        self.active_parent_indices = self.all_parent_indices[positions]
        self.categories = self.sample_category[self.active_parent_indices]
        self.num_windows = int(len(self.active_parent_indices))
        self.num_steps = int(len(self.active_parent_indices))

    def set_all_active(self):
        self.active_parent_indices = self.all_parent_indices.copy()
        self.categories = self.all_categories.copy()
        self.num_windows = int(len(self.active_parent_indices))
        self.num_steps = int(len(self.active_parent_indices))

    def subset_by_category(self, category_id, name, active=False):
        if active:
            parent_indices = self.active_parent_indices[
                self.sample_category[self.active_parent_indices] == int(category_id)
            ]
        else:
            positions = np.flatnonzero(self.all_categories == int(category_id))
            parent_indices = self.all_parent_indices[positions]
        return AeroTAFATTNWindowDataset(
            self.path,
            history_windows=self.history_windows,
            raw_cache_size=self.raw_cache_size,
            parent_indices=parent_indices,
            name=name,
            shared_target=self,
        )

    def category_counts(self, active=False):
        categories = self.categories if active else self.all_categories
        return {name: int(np.sum(categories == idx)) for idx, name in enumerate(CATEGORY_NAMES)}


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
    obs_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(int(dataset.obs_dim),), dtype=np.float32)
    act_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(int(dataset.act_dim),), dtype=np.float32)
    return obs_space, act_space


class StepPartitionBatchSampler:
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


def collate_window_batch(samples):
    return samples


def build_train_loader(dataset, args, device, epoch):
    return DataLoader(
        dataset,
        batch_sampler=StepPartitionBatchSampler(
            dataset_size=len(dataset),
            num_batches=args.effective_mini_epoch,
            seed=args.seed,
            epoch=epoch,
            shuffle=True,
        ),
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_window_batch,
    )


def build_eval_loader(dataset, args, device):
    return DataLoader(
        dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
        collate_fn=collate_window_batch,
    )


def compute_r_value(sse, target_sum, target_square_sum, count):
    if count <= 0:
        return 0.0
    sst = target_square_sum - target_sum * target_sum / count
    if sst <= 1e-12:
        return 0.0
    return 1.0 - sse / sst


def new_totals():
    return {
        "loss": 0.0,
        "threat_loss": 0.0,
        "attack_loss": 0.0,
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


def update_totals(totals, loss, threat_loss, attack_loss, threat_pred, attack_pred, threat_targets, attack_targets, count):
    totals["loss"] += float(loss.item()) * count
    totals["threat_loss"] += float(threat_loss.item()) * count
    totals["attack_loss"] += float(attack_loss.item()) * count
    totals["steps"] += int(count)

    threat_error = threat_pred - threat_targets
    attack_error = attack_pred - attack_targets
    totals["threat_sse"] += torch.sum(threat_error * threat_error).item()
    totals["attack_sse"] += torch.sum(attack_error * attack_error).item()
    totals["threat_sum"] += torch.sum(threat_targets).item()
    totals["attack_sum"] += torch.sum(attack_targets).item()
    totals["threat_square_sum"] += torch.sum(threat_targets * threat_targets).item()
    totals["attack_square_sum"] += torch.sum(attack_targets * attack_targets).item()
    totals["threat_count"] += threat_targets.numel()
    totals["attack_count"] += attack_targets.numel()


def finalize_metrics(totals):
    steps = max(int(totals["steps"]), 1)
    return {
        "loss": totals["loss"] / steps,
        "threat_loss": totals["threat_loss"] / steps,
        "attack_loss": totals["attack_loss"] / steps,
        "steps": int(totals["steps"]),
        "threat_r": compute_r_value(totals["threat_sse"], totals["threat_sum"], totals["threat_square_sum"], totals["threat_count"]),
        "attack_r": compute_r_value(totals["attack_sse"], totals["attack_sum"], totals["attack_square_sum"], totals["attack_count"]),
    }


def forward_losses(model, batch, device, args):
    threat_preds = []
    attack_preds = []
    threat_targets = []
    attack_targets = []

    for obs_seq, action_seq, threat_target, attack_target, time_offset in batch:
        obs_seq = torch.as_tensor(obs_seq, dtype=torch.float32, device=device)
        action_seq = torch.as_tensor(action_seq, dtype=torch.float32, device=device)
        seq_len = int(obs_seq.shape[0])
        obs_flat = obs_seq.reshape(seq_len * obs_seq.shape[1], -1)
        action_flat = action_seq.reshape(seq_len * action_seq.shape[1], -1)

        threat_output, attack_output = model(
            obs_flat,
            action_flat,
            seq_len=seq_len,
            time_offset=int(time_offset),
        )
        threat_preds.append(threat_output[-1])
        attack_preds.append(attack_output[-1])
        threat_targets.append(torch.as_tensor(threat_target, dtype=torch.float32, device=device).reshape(-1))
        attack_targets.append(torch.as_tensor(attack_target, dtype=torch.float32, device=device).reshape(-1))

    threat_pred = torch.stack(threat_preds, dim=0).reshape(len(batch), -1)
    attack_pred = torch.stack(attack_preds, dim=0).reshape(len(batch), -1)
    threat_targets = torch.stack(threat_targets, dim=0).reshape(len(batch), -1)
    attack_targets = torch.stack(attack_targets, dim=0).reshape(len(batch), -1)

    raw_threat_loss = F.mse_loss(threat_pred, threat_targets)
    raw_attack_loss = F.mse_loss(attack_pred, attack_targets)
    loss = args.threat_loss_weight * raw_threat_loss + args.attack_loss_weight * raw_attack_loss
    return loss, raw_threat_loss, raw_attack_loss, threat_pred, attack_pred, threat_targets, attack_targets, len(batch)


def train_one_epoch(model, loader, device, args, optimizer):
    model.train()
    totals = new_totals()
    update_count = 0
    update_total = len(loader)
    for batch in loader:
        start_time = time.time()
        update_count += 1
        outputs = forward_losses(model, batch, device, args)
        loss, raw_threat_loss, raw_attack_loss, threat_pred, attack_pred, threat_targets, attack_targets, count = outputs

        optimizer.zero_grad()
        loss.backward()
        if args.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()

        weighted_threat = args.threat_loss_weight * raw_threat_loss.item()
        weighted_attack = args.attack_loss_weight * raw_attack_loss.item()
        elapsed = time.time() - start_time
        logging.info(
            f"  [U{update_count:04d}/{update_total:04d}] "
            f"loss={loss.item():.5f} "
            f"| raw(th/a)=({raw_threat_loss.item():.5f}/{raw_attack_loss.item():.5f}) "
            f"| weighted(th/a)=({weighted_threat:.5f}/{weighted_attack:.5f}) "
            f"| windows={count} | {elapsed:.2f}s"
        )
        update_totals(totals, loss, raw_threat_loss, raw_attack_loss, threat_pred, attack_pred, threat_targets, attack_targets, count)

    metrics = finalize_metrics(totals)
    metrics["updates"] = int(update_count)
    return metrics


def evaluate(model, loader, device, args):
    model.eval()
    totals = new_totals()
    with torch.no_grad():
        for batch in loader:
            outputs = forward_losses(model, batch, device, args)
            loss, raw_threat_loss, raw_attack_loss, threat_pred, attack_pred, threat_targets, attack_targets, count = outputs
            update_totals(totals, loss, raw_threat_loss, raw_attack_loss, threat_pred, attack_pred, threat_targets, attack_targets, count)
    return finalize_metrics(totals)


def append_csv_row(path, row, write_header=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def save_checkpoint(path, model, optimizer, epoch, val_loss, args):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
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
        "all_target_path": normalize_path(datasets["train"].all_target_path),
        "dataset_summary": {
            key: {
                "points": int(len(dataset.all_parent_indices)),
                "active_points": int(len(dataset)),
                "category_counts": dataset.category_counts(active=False),
                "obs_dim": int(dataset.obs_dim),
                "act_dim": int(dataset.act_dim),
                "num_agents": int(dataset.num_agents),
                "history_windows": int(dataset.history_windows),
            }
            for key, dataset in datasets.items()
        },
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def format_counts(counts):
    return "{" + ", ".join(f"{name}={int(counts.get(name, 0))}" for name in CATEGORY_NAMES) + "}"


def evaluate_raw_losses_by_category(model, dataset, device, args, active=True):
    results = {}
    old_eval_batch_size = args.eval_batch_size
    try:
        for category_id, category_name in enumerate(CATEGORY_NAMES):
            if active:
                category_parent_indices = dataset.active_parent_indices[
                    dataset.sample_category[dataset.active_parent_indices] == int(category_id)
                ]
            else:
                positions = np.flatnonzero(dataset.all_categories == int(category_id))
                category_parent_indices = dataset.all_parent_indices[positions]

            if category_parent_indices.size == 0:
                results[category_name] = {"threat_loss": float("nan"), "attack_loss": float("nan")}
                continue

            category_dataset = AeroTAFATTNWindowDataset(
                dataset.path,
                history_windows=dataset.history_windows,
                raw_cache_size=dataset.raw_cache_size,
                parent_indices=category_parent_indices,
                name=f"{dataset.name}_{category_name}",
                shared_target=dataset,
            )
            category_dataset.set_all_active()
            args.eval_batch_size = max(1, min(old_eval_batch_size, len(category_dataset)))
            metrics = evaluate(model, build_eval_loader(category_dataset, args, device), device, args)
            results[category_name] = {
                "threat_loss": metrics["threat_loss"],
                "attack_loss": metrics["attack_loss"],
            }
    finally:
        args.eval_batch_size = old_eval_batch_size
    return results


def evaluate_test_by_category(model, test_dataset, device, args):
    results = {}
    test_dataset.set_all_active()
    eval_batch_size = max(1, int(np.ceil(len(test_dataset) / max(1, int(args.mini_epoch)))))
    results["all"] = evaluate(model, build_eval_loader(test_dataset, args, device), device, args)

    for category_id, category_name in enumerate(CATEGORY_NAMES):
        positions = np.flatnonzero(test_dataset.all_categories == int(category_id))
        if positions.size == 0:
            results[category_name] = finalize_metrics(new_totals())
            continue
        category_dataset = test_dataset.subset_by_category(category_id, f"test_{category_name}")
        category_dataset.set_all_active()
        old_eval_batch_size = args.eval_batch_size
        args.eval_batch_size = max(1, min(eval_batch_size, len(category_dataset)))
        results[category_name] = evaluate(model, build_eval_loader(category_dataset, args, device), device, args)
        args.eval_batch_size = old_eval_batch_size
    return results


def add_category_raw_loss_columns(row, prefix, category_losses):
    for category_name in CATEGORY_NAMES:
        row[f"{prefix}_{category_name}_threat_loss"] = f"{category_losses[category_name]['threat_loss']:.8f}"
        row[f"{prefix}_{category_name}_attack_loss"] = f"{category_losses[category_name]['attack_loss']:.8f}"


def get_parser():
    parser = argparse.ArgumentParser(description="Train AeroTAF_ATTN on detailed point-index datasets.")
    parser.add_argument("--dataset-dir", type=str, required=True, help="Processed dataset directory.")
    parser.add_argument("--train-file", type=str, default="train.npz", help="Training split filename.")
    parser.add_argument("--val-file", type=str, default="val.npz", help="Validation split filename.")
    parser.add_argument("--test-file", type=str, default="test.npz", help="Test split filename.")
    parser.add_argument("--experiment-name", type=str, default="AeroTAF-ATTN-Detail-Point-K100", help="Experiment name.")
    parser.add_argument("--save-root", type=str, default="scripts/results/AeroTAF", help="Checkpoint root directory.")
    parser.add_argument("--seed", type=int, default=1, help="Random seed.")
    parser.add_argument("--cuda", action="store_true", default=False, help="Use CUDA if available.")
    parser.add_argument("--cuda-device-id", type=int, default=0, help="CUDA device id.")
    parser.add_argument("--n-training-threads", type=int, default=1, help="Torch training thread count.")
    parser.add_argument("--mini-epoch", type=int, default=100, help="Number of parameter updates per epoch.")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs.")
    parser.add_argument("--stable-sample-ratio", type=float, default=0.05, help="Stable samples used per train/val epoch.")
    parser.add_argument("--history-windows", type=int, default=16, help="Maximum number of historical timesteps per sample.")
    parser.add_argument("--raw-cache-size", type=int, default=4, help="Number of raw episode npz files cached per dataset instance.")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Adam weight decay.")
    parser.add_argument("--max-grad-norm", type=float, default=2.0, help="Gradient clipping max norm.")
    parser.add_argument("--threat-loss-weight", type=float, default=1.0, help="Weight for threat regression loss.")
    parser.add_argument("--attack-loss-weight", type=float, default=8.0, help="Weight for attack regression loss.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker count.")
    parser.add_argument("--save-interval", type=int, default=10, help="Latest checkpoint save interval in epochs.")
    parser.add_argument("--log-interval", type=int, default=1, help="Epoch logging interval.")
    parser.add_argument("--num-agents", type=int, default=4, help="Number of ego agents.")
    parser.add_argument("--activation-id", type=int, default=1, help="0:Tanh, 1:ReLU, 2:LeakyReLU, 3:ELU.")
    parser.add_argument("--use-feature-normalization", action="store_true", default=False, help="Use LayerNorm on obs/action inputs.")
    parser.add_argument("--KQ-hidden-size", type=str, default="128 128", help="K/Q MLP hidden sizes.")
    parser.add_argument("--V-hidden-size", type=str, default="128 128", help="V MLP hidden sizes.")
    parser.add_argument("--attn-output-hidden-size", type=str, default="64 32", help="Temporal FFN hidden sizes.")
    parser.add_argument("--field-output-hidden-size", type=str, default="64 32", help="Threat/attack output head hidden sizes.")
    parser.add_argument("--num-heads", type=int, default=4, help="Spatial attention head count.")
    parser.add_argument("--time-head-num", type=int, default=4, help="Temporal attention head count.")
    return parser


def main(args):
    parser = get_parser()
    all_args = parser.parse_args(args)

    if all_args.history_windows <= 0:
        raise ValueError("--history-windows must be positive")

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
        "train": AeroTAFATTNWindowDataset(
            dataset_paths["train"],
            history_windows=all_args.history_windows,
            raw_cache_size=all_args.raw_cache_size,
        ),
        "val": AeroTAFATTNWindowDataset(
            dataset_paths["val"],
            history_windows=all_args.history_windows,
            raw_cache_size=all_args.raw_cache_size,
        ),
        "test": AeroTAFATTNWindowDataset(
            dataset_paths["test"],
            history_windows=all_args.history_windows,
            raw_cache_size=all_args.raw_cache_size,
        ),
    }
    if datasets["train"].num_agents != all_args.num_agents:
        raise ValueError(f"--num-agents={all_args.num_agents}, but raw data has {datasets['train'].num_agents} agents")
    all_args.effective_mini_epoch = max(1, min(int(all_args.mini_epoch), len(datasets["train"])))
    all_args.eval_batch_size = max(1, int(np.ceil(len(datasets["train"]) / all_args.effective_mini_epoch)))

    obs_space, act_space = build_spaces(datasets["train"])
    model = PPOAeroTAFATTN(AeroTAFATTNArgs(all_args), obs_space, act_space, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=all_args.lr, weight_decay=all_args.weight_decay)
    run_dir = build_run_dir(all_args)
    run_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = run_dir / "AeroTAF_ATTN_best.pt"
    latest_ckpt_path = run_dir / "AeroTAF_ATTN_latest.pt"
    epoch_log_path = run_dir / "epoch_log.csv"
    config_path = run_dir / "config.json"
    test_metrics_path = run_dir / "test_metrics.json"
    dump_config(config_path, all_args, dataset_paths, datasets)

    logging.info("=" * 72)
    logging.info("AeroTAF_ATTN Detail Window Training")
    logging.info("=" * 72)
    logging.info(f"device      : {device}")
    logging.info(f"dataset     : {normalize_path(dataset_dir)}")
    logging.info(f"all target  : {normalize_path(datasets['train'].all_target_path)}")
    logging.info(f"run dir     : {normalize_path(run_dir)}")
    logging.info(f"history     : {all_args.history_windows}")
    logging.info(f"dims        : agents={datasets['train'].num_agents} obs={datasets['train'].obs_dim} action={datasets['train'].act_dim}")
    logging.info(
        f"split rows  : train={len(datasets['train'].all_parent_indices)} "
        f"val={len(datasets['val'].all_parent_indices)} test={len(datasets['test'].all_parent_indices)} "
        f"| stable_ratio={all_args.stable_sample_ratio}"
    )
    logging.info(f"train cats  : {format_counts(datasets['train'].category_counts(active=False))}")
    logging.info(f"val cats    : {format_counts(datasets['val'].category_counts(active=False))}")
    logging.info(f"test cats   : {format_counts(datasets['test'].category_counts(active=False))}")
    logging.info(
        f"loss w      : threat={all_args.threat_loss_weight} attack={all_args.attack_loss_weight} | lr={all_args.lr:.2e}"
    )
    logging.info("-" * 72)

    best_val_loss = float("inf")
    for epoch in range(1, all_args.epochs + 1):
        start_time = time.time()
        datasets["train"].set_epoch_sample(all_args.stable_sample_ratio, all_args.seed, epoch, shuffle=True)
        datasets["val"].set_epoch_sample(all_args.stable_sample_ratio, all_args.seed + 100000, epoch, shuffle=True)
        all_args.effective_mini_epoch = max(1, min(int(all_args.mini_epoch), len(datasets["train"])))
        all_args.eval_batch_size = max(1, int(np.ceil(len(datasets["train"]) / all_args.effective_mini_epoch)))

        logging.info(
            f"[E{epoch:03d}/{all_args.epochs:03d}] "
            f"train_active={len(datasets['train'])} {format_counts(datasets['train'].category_counts(active=True))} | "
            f"val_active={len(datasets['val'])} {format_counts(datasets['val'].category_counts(active=True))} | "
            f"mini_epoch={all_args.effective_mini_epoch} | windows/update~{all_args.eval_batch_size}"
        )

        train_loader = build_train_loader(datasets["train"], all_args, device, epoch)
        train_metrics = train_one_epoch(model, train_loader, device, all_args, optimizer)
        val_loader = build_eval_loader(datasets["val"], all_args, device)
        val_metrics = evaluate(model, val_loader, device, all_args)
        train_category_losses = evaluate_raw_losses_by_category(model, datasets["train"], device, all_args, active=True)
        val_category_losses = evaluate_raw_losses_by_category(model, datasets["val"], device, all_args, active=True)
        elapsed = time.time() - start_time

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            save_checkpoint(best_ckpt_path, model, optimizer, epoch, best_val_loss, all_args)
            logging.info(f"[BEST] epoch={epoch:03d} val={best_val_loss:.6f} -> {normalize_path(best_ckpt_path)}")

        if epoch % all_args.save_interval == 0 or epoch == all_args.epochs:
            save_checkpoint(latest_ckpt_path, model, optimizer, epoch, best_val_loss, all_args)

        csv_row = {
            "epoch": epoch,
            "train_raw_threat_loss": f"{train_metrics['threat_loss']:.8f}",
            "train_raw_attack_loss": f"{train_metrics['attack_loss']:.8f}",
            "val_raw_threat_loss": f"{val_metrics['threat_loss']:.8f}",
            "val_raw_attack_loss": f"{val_metrics['attack_loss']:.8f}",
        }
        add_category_raw_loss_columns(csv_row, "train", train_category_losses)
        add_category_raw_loss_columns(csv_row, "val", val_category_losses)
        append_csv_row(epoch_log_path, csv_row, write_header=(epoch == 1))

        if epoch % all_args.log_interval == 0 or epoch == 1 or epoch == all_args.epochs:
            logging.info(
                f"[VAL {epoch:03d}] "
                f"train={train_metrics['loss']:.5f} "
                f"| val={val_metrics['loss']:.5f} "
                f"| raw_val(th/a)=({val_metrics['threat_loss']:.5f}/{val_metrics['attack_loss']:.5f}) "
                f"| R(th/a)=({val_metrics['threat_r']:.3f}/{val_metrics['attack_r']:.3f}) "
                f"| {elapsed:.1f}s"
            )

    if best_ckpt_path.exists():
        checkpoint = torch.load(best_ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        logging.info(f"loaded best checkpoint for final test: {normalize_path(best_ckpt_path)}")

    test_metrics = evaluate_test_by_category(model, datasets["test"], device, all_args)
    with open(test_metrics_path, "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2, ensure_ascii=False)

    logging.info("-" * 72)
    for name in ["all"] + list(CATEGORY_NAMES):
        metrics = test_metrics[name]
        logging.info(
            f"[TEST:{name:11s}] loss={metrics['loss']:.6f} "
            f"| raw(th/a)=({metrics['threat_loss']:.6f}/{metrics['attack_loss']:.6f}) "
            f"| R(th/a)=({metrics['threat_r']:.4f}/{metrics['attack_r']:.4f}) "
            f"| windows={metrics['steps']}"
        )
    logging.info(f"test metrics saved: {normalize_path(test_metrics_path)}")
    logging.info("Done.")


if __name__ == "__main__":
    default_args = [
        "--dataset-dir", "datasets/aerotaf/4v4_shoot_mappo_pool/fkr-300vs500/processed_detail_index_k_target_K100",
        "--experiment-name", "AeroTAF-ATTN-Detail-Point-K100",
        "--save-root", "scripts/results/AeroTAF",
        "--seed", "1",
        "--n-training-threads", "1",
        "--mini-epoch", "10",
        "--epochs", "10",
        "--stable-sample-ratio", "0.05",
        "--history-windows", "100",
        "--raw-cache-size", "4",
        "--lr", "3e-5",
        "--weight-decay", "1e-4",
        "--max-grad-norm", "2.0",
        "--threat-loss-weight", "1.0",
        "--attack-loss-weight", "2.0",
        "--num-workers", "0",
        "--save-interval", "10",
        "--log-interval", "1",
        "--num-agents", "4",
        "--activation-id", "1",
        "--KQ-hidden-size", "128 128",
        "--V-hidden-size", "128 128",
        "--attn-output-hidden-size", "64 32",
        "--field-output-hidden-size", "64 32",
        "--num-heads", "4",
        "--time-head-num", "4",
        "--use-feature-normalization",
        # "--cuda",
        # "--cuda-device-id", "0",
    ]
    main(sys.argv[1:] if len(sys.argv) > 1 else default_args)
