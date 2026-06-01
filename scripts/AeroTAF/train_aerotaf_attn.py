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

try:
    import gymnasium as gym
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset, Subset
except ModuleNotFoundError as exc:
    print(f"Error: missing dependency: {exc}")
    print("Please activate the same Python environment used by this project, then run this script again.")
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


class ProcessedSplitStore:
    def __init__(self, npz_path, sample_stride=1):
        if sample_stride < 1:
            raise ValueError(f"sample_stride must be >= 1, got {sample_stride}")

        with np.load(npz_path, allow_pickle=True) as data:
            required_keys = ("obs", "actions", "threat_targets", "attack_targets", "episode_lengths")
            missing = [key for key in required_keys if key not in data.files]
            if missing:
                raise KeyError(f"{npz_path} missing keys: {missing}")

            self.obs = data["obs"].astype(np.float32, copy=False)
            self.actions = data["actions"].astype(np.float32, copy=False)
            self.threat_targets = data["threat_targets"].astype(np.float32, copy=False)
            self.attack_targets = data["attack_targets"].astype(np.float32, copy=False)
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
        return (
            episode["obs"][start:end],
            episode["actions"][start:end],
            episode["threat_targets"][start:end],
            episode["attack_targets"][start:end],
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


def build_train_subset_dataset(dataset, train_mini_batches, epoch, seed):
    total_chunks = len(dataset)
    if total_chunks == 0:
        raise ValueError("Empty training chunk dataset.")

    if train_mini_batches <= 1 or total_chunks == 1:
        return dataset, 1, 1, total_chunks

    effective_batches = min(int(train_mini_batches), total_chunks)
    cycle_index = (epoch - 1) % effective_batches
    cycle_round = (epoch - 1) // effective_batches

    rng = random.Random(seed + cycle_round * 9973)
    indices = list(range(total_chunks))
    rng.shuffle(indices)
    split_indices = np.array_split(np.asarray(indices, dtype=np.int64), effective_batches)
    selected = split_indices[cycle_index].tolist()
    if not selected:
        raise RuntimeError(
            f"Empty train mini-batch split detected: total_chunks={total_chunks}, "
            f"train_mini_batches={train_mini_batches}, effective_batches={effective_batches}, epoch={epoch}"
        )
    return Subset(dataset, selected), cycle_index + 1, effective_batches, len(selected)


def build_epoch_subset_dataset(dataset, split_mini_batches, epoch, seed, seed_offset=0):
    total_chunks = len(dataset)
    if total_chunks == 0:
        raise ValueError("Empty chunk dataset.")

    if split_mini_batches <= 1 or total_chunks == 1:
        return dataset, 1, 1, total_chunks

    effective_batches = min(int(split_mini_batches), total_chunks)
    cycle_index = (epoch - 1) % effective_batches
    cycle_round = (epoch - 1) // effective_batches

    rng = random.Random(seed + seed_offset + cycle_round * 9973)
    indices = list(range(total_chunks))
    rng.shuffle(indices)
    split_indices = np.array_split(np.asarray(indices, dtype=np.int64), effective_batches)
    selected = split_indices[cycle_index].tolist()
    if not selected:
        raise RuntimeError(
            f"Empty split mini-batch detected: total_chunks={total_chunks}, "
            f"split_mini_batches={split_mini_batches}, effective_batches={effective_batches}, epoch={epoch}"
        )
    return Subset(dataset, selected), cycle_index + 1, effective_batches, len(selected)


def prepare_episode_batch(obs, actions, threat_targets, attack_targets, seq_len, time_offset, device):
    seq_len = int(seq_len)
    time_offset = int(time_offset)

    obs = torch.as_tensor(obs, dtype=torch.float32, device=device).reshape(seq_len, obs.shape[1], -1)
    actions = torch.as_tensor(actions, dtype=torch.float32, device=device).reshape(seq_len, actions.shape[1], -1)
    threat_targets = torch.as_tensor(threat_targets, dtype=torch.float32, device=device).reshape(seq_len, -1)
    attack_targets = torch.as_tensor(attack_targets, dtype=torch.float32, device=device).reshape(seq_len, -1)

    obs = obs.reshape(seq_len * obs.shape[1], -1)
    actions = actions.reshape(seq_len * actions.shape[1], -1)
    return obs, actions, threat_targets, attack_targets, seq_len, time_offset


def evaluate(model, data_loader, device):
    model.eval()
    total_loss = 0.0
    total_threat_loss = 0.0
    total_attack_loss = 0.0
    total_steps = 0

    with torch.no_grad():
        for obs, actions, threat_targets, attack_targets, seq_len, time_offset in data_loader:
            obs, actions, threat_targets, attack_targets, seq_len, time_offset = prepare_episode_batch(
                obs,
                actions,
                threat_targets,
                attack_targets,
                seq_len,
                time_offset,
                device,
            )
            _, threat_pred, attack_pred = model(obs, actions, seq_len=seq_len, time_offset=time_offset)

            threat_loss = F.mse_loss(threat_pred, threat_targets)
            attack_loss = F.mse_loss(attack_pred, attack_targets)
            loss = threat_loss + attack_loss

            total_loss += loss.item() * seq_len
            total_threat_loss += threat_loss.item() * seq_len
            total_attack_loss += attack_loss.item() * seq_len
            total_steps += seq_len

    return {
        "loss": total_loss / max(total_steps, 1),
        "threat_loss": total_threat_loss / max(total_steps, 1),
        "attack_loss": total_attack_loss / max(total_steps, 1),
        "valid_steps": int(total_steps),
    }


def train_one_epoch(model, optimizer, data_loader, device, max_grad_norm):
    model.train()
    total_loss = 0.0
    total_threat_loss = 0.0
    total_attack_loss = 0.0
    total_steps = 0

    for obs, actions, threat_targets, attack_targets, seq_len, time_offset in data_loader:
        obs, actions, threat_targets, attack_targets, seq_len, time_offset = prepare_episode_batch(
            obs,
            actions,
            threat_targets,
            attack_targets,
            seq_len,
            time_offset,
            device,
        )
        _, threat_pred, attack_pred = model(obs, actions, seq_len=seq_len, time_offset=time_offset)

        threat_loss = F.mse_loss(threat_pred, threat_targets)
        attack_loss = F.mse_loss(attack_pred, attack_targets)
        loss = threat_loss + attack_loss

        optimizer.zero_grad()
        loss.backward()
        if max_grad_norm is not None and max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        total_loss += loss.item() * seq_len
        total_threat_loss += threat_loss.item() * seq_len
        total_attack_loss += attack_loss.item() * seq_len
        total_steps += seq_len

    return {
        "loss": total_loss / max(total_steps, 1),
        "threat_loss": total_threat_loss / max(total_steps, 1),
        "attack_loss": total_attack_loss / max(total_steps, 1),
        "valid_steps": int(total_steps),
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


def save_checkpoint(path, model, optimizer, epoch, best_val_loss, args):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
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
            "train_mini_batches": int(args.train_mini_batches),
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
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker count.")
    parser.add_argument("--save-interval", type=int, default=10, help="Latest checkpoint save interval.")
    parser.add_argument("--log-interval", type=int, default=1, help="Epoch logging interval.")
    parser.add_argument("--chunk-length", type=int, default=32, help="Sliding-window chunk length on the time axis.")
    parser.add_argument("--chunk-stride", type=int, default=8, help="Sliding-window stride on the time axis.")
    parser.add_argument(
        "--train-mini-batches",
        type=int,
        default=1,
        help="Split all train chunks into this many partitions and train one partition per epoch. 1 uses all train chunks each epoch.",
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
        "train": ProcessedSplitStore(dataset_paths["train"], sample_stride=all_args.sample_stride),
        "val": ProcessedSplitStore(dataset_paths["val"], sample_stride=all_args.sample_stride),
        "test": ProcessedSplitStore(dataset_paths["test"], sample_stride=all_args.sample_stride),
    }

    obs_space, act_space = build_spaces(stores["train"])
    model_args = AeroTAFATTNArgs(all_args)
    model = PPOAeroTAFATTN(model_args, obs_space, act_space, device=device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=all_args.lr,
        weight_decay=all_args.weight_decay,
    )

    datasets = {
        "train": AeroTAFATTNChunkDataset(stores["train"], all_args.chunk_length, all_args.chunk_stride),
        "val": AeroTAFATTNChunkDataset(stores["val"], all_args.chunk_length, all_args.chunk_stride),
        "test": AeroTAFATTNChunkDataset(stores["test"], all_args.chunk_length, all_args.chunk_stride),
    }

    loaders = {
        "test_full": DataLoader(
            datasets["test"],
            batch_size=all_args.batch_size,
            shuffle=False,
            num_workers=all_args.num_workers,
            pin_memory=(device.type == "cuda"),
            collate_fn=episode_collate_fn,
        ),
    }

    run_dir = build_run_dir(all_args)
    run_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = run_dir / "AeroTAF_ATTN_best.pt"
    latest_ckpt_path = run_dir / "AeroTAF_ATTN_latest.pt"
    log_path = run_dir / "train_log.csv"
    config_path = run_dir / "config.json"
    test_metrics_path = run_dir / "test_metrics.json"
    dump_config(config_path, all_args, dataset_paths, stores, datasets)

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
        f"train mini-batches per full train-chunk cycle: {max(1, min(all_args.train_mini_batches, len(datasets['train'])))}"
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

    best_val_loss = float("inf")

    for epoch in range(1, all_args.epochs + 1):
        start_time = time.time()

        train_subset, mini_batch_index, effective_mini_batches, train_chunks_used = build_train_subset_dataset(
            datasets["train"],
            all_args.train_mini_batches,
            epoch,
            all_args.seed,
        )
        val_subset, val_mini_batch_index, val_effective_mini_batches, val_chunks_used = build_epoch_subset_dataset(
            datasets["val"],
            all_args.train_mini_batches,
            epoch,
            all_args.seed,
            seed_offset=100000,
        )
        train_loader = DataLoader(
            train_subset,
            batch_size=all_args.batch_size,
            shuffle=True,
            num_workers=all_args.num_workers,
            pin_memory=(device.type == "cuda"),
            collate_fn=episode_collate_fn,
        )
        val_loader = DataLoader(
            val_subset,
            batch_size=all_args.batch_size,
            shuffle=False,
            num_workers=all_args.num_workers,
            pin_memory=(device.type == "cuda"),
            collate_fn=episode_collate_fn,
        )

        train_metrics = train_one_epoch(
            model=model,
            optimizer=optimizer,
            data_loader=train_loader,
            device=device,
            max_grad_norm=all_args.max_grad_norm,
        )
        val_metrics = evaluate(model=model, data_loader=val_loader, device=device)
        elapsed = time.time() - start_time

        row = {
            "epoch": epoch,
            "train_loss": f"{train_metrics['loss']:.8f}",
            "train_threat_loss": f"{train_metrics['threat_loss']:.8f}",
            "train_attack_loss": f"{train_metrics['attack_loss']:.8f}",
            "val_loss": f"{val_metrics['loss']:.8f}",
            "val_threat_loss": f"{val_metrics['threat_loss']:.8f}",
            "val_attack_loss": f"{val_metrics['attack_loss']:.8f}",
            "mini_batch_index": mini_batch_index,
            "mini_batch_total": effective_mini_batches,
            "val_mini_batch_index": val_mini_batch_index,
            "val_mini_batch_total": val_effective_mini_batches,
            "train_valid_steps": train_metrics["valid_steps"],
            "val_valid_steps": val_metrics["valid_steps"],
            "train_chunks_used": train_chunks_used,
            "train_chunks_total": len(datasets["train"]),
            "val_chunks_used": val_chunks_used,
            "val_chunks_total": len(datasets["val"]),
            "time_sec": f"{elapsed:.4f}",
        }
        append_csv_row(log_path, row, write_header=(epoch == 1))

        if epoch % all_args.log_interval == 0 or epoch == 1 or epoch == all_args.epochs:
            logging.info(
                f"epoch {epoch:03d} | "
                f"mini-batch {mini_batch_index:02d}/{effective_mini_batches:02d} "
                f"(chunks={train_chunks_used}/{len(datasets['train'])}) | "
                f"val-batch {val_mini_batch_index:02d}/{val_effective_mini_batches:02d} "
                f"(chunks={val_chunks_used}/{len(datasets['val'])}) | "
                f"train loss={train_metrics['loss']:.6f} "
                f"(threat={train_metrics['threat_loss']:.6f}, attack={train_metrics['attack_loss']:.6f}) | "
                f"val loss={val_metrics['loss']:.6f} "
                f"(threat={val_metrics['threat_loss']:.6f}, attack={val_metrics['attack_loss']:.6f}) | "
                f"time={elapsed:.2f}s"
            )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            save_checkpoint(best_ckpt_path, model, optimizer, epoch, best_val_loss, all_args)
            logging.info(f"best checkpoint updated: {normalize_path(best_ckpt_path)}")

        if epoch % all_args.save_interval == 0 or epoch == all_args.epochs:
            save_checkpoint(latest_ckpt_path, model, optimizer, epoch, best_val_loss, all_args)
            logging.info(f"latest checkpoint saved: {normalize_path(latest_ckpt_path)}")

    if best_ckpt_path.exists():
        checkpoint = torch.load(best_ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        logging.info(f"loaded best checkpoint for final test: {normalize_path(best_ckpt_path)}")

    test_metrics = evaluate(model=model, data_loader=loaders["test_full"], device=device)
    with open(test_metrics_path, "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2, ensure_ascii=False)

    logging.info(
        f"test loss={test_metrics['loss']:.6f} "
        f"(threat={test_metrics['threat_loss']:.6f}, attack={test_metrics['attack_loss']:.6f})"
    )
    logging.info(f"test metrics saved: {normalize_path(test_metrics_path)}")
    logging.info("Done.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    default_args = [
        "--dataset-dir", "datasets/aerotaf/4v4_shoot_mappo_pool_stage1/processed_stage1_K20",
        "--experiment-name", "AeroTAF-ATTN-Stage1-K20-SlidingWindow",
        "--save-root", "scripts/results/AeroTAF_ATTN",
        "--seed", "1",
        "--n-training-threads", "1",
        "--batch-size", "1",
        "--epochs", "50",
        "--lr", "1e-4",
        "--weight-decay", "1e-4",
        "--max-grad-norm", "2.0",
        "--num-workers", "0",
        "--save-interval", "10",
        "--log-interval", "1",
        "--chunk-length", "50",
        "--chunk-stride", "30",
        "--train-mini-batches", "10",
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
    main(sys.argv[1:] if len(sys.argv) > 1 else default_args)
