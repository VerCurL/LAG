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
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset, Subset
except ModuleNotFoundError as exc:
    print(f"Error: missing dependency: {exc}")
    print("Please activate the same Python environment used by this project, then run this script again.")
    sys.exit(1)

from algorithms.mappoCFC.ppo_AeroTAF import PPOAeroTAF
from scripts.AeroTAF.collector.path_utils import normalize_path, resolve_project_path


class AeroTAFArgs:
    def __init__(self, args):
        self.num_agents = args.num_agents
        self.activation_id = args.activation_id
        self.use_feature_normalization = args.use_feature_normalization
        self.num_heads = args.num_heads
        self.KQ_hidden_size = args.KQ_hidden_size
        self.V_hidden_size = args.V_hidden_size
        self.AeroTAF_out_hidden_size = args.AeroTAF_out_hidden_size


class AeroTAFDataset(Dataset):
    def __init__(self, npz_path, sample_stride=1):
        if sample_stride < 1:
            raise ValueError(f"sample_stride must be >= 1, got {sample_stride}")

        with np.load(npz_path, allow_pickle=True) as data:
            required_keys = ("obs", "actions", "threat_targets", "attack_targets")
            missing = [key for key in required_keys if key not in data.files]
            if missing:
                raise KeyError(f"{npz_path} missing keys: {missing}")

            obs = data["obs"].astype(np.float32, copy=False)
            actions = data["actions"].astype(np.float32, copy=False)
            threat_targets = data["threat_targets"].astype(np.float32, copy=False)
            attack_targets = data["attack_targets"].astype(np.float32, copy=False)
            episode_lengths = data["episode_lengths"].astype(np.int32, copy=False) if "episode_lengths" in data.files else None

        if obs.shape[0] != actions.shape[0]:
            raise ValueError(f"{npz_path}: obs/actions sample count mismatch")
        if obs.shape[0] != threat_targets.shape[0]:
            raise ValueError(f"{npz_path}: obs/threat_targets sample count mismatch")
        if obs.shape[0] != attack_targets.shape[0]:
            raise ValueError(f"{npz_path}: obs/attack_targets sample count mismatch")

        self.original_sample_count = int(obs.shape[0])
        self.sample_stride = int(sample_stride)

        if self.sample_stride == 1:
            keep_indices = None
        else:
            if episode_lengths is None:
                raise KeyError(
                    f"{npz_path}: episode_lengths is required when sample_stride > 1"
                )

            keep_indices = []
            start = 0
            for length in episode_lengths.tolist():
                end = start + int(length)
                keep_indices.extend(range(start, end, self.sample_stride))
                start = end

            keep_indices = np.asarray(keep_indices, dtype=np.int64)

        if keep_indices is None:
            self.obs = obs
            self.actions = actions
            self.threat_targets = threat_targets
            self.attack_targets = attack_targets
        else:
            self.obs = obs[keep_indices]
            self.actions = actions[keep_indices]
            self.threat_targets = threat_targets[keep_indices]
            self.attack_targets = attack_targets[keep_indices]

        self.kept_sample_count = int(self.obs.shape[0])

    def __len__(self):
        return self.obs.shape[0]

    def __getitem__(self, index):
        return (
            self.obs[index],
            self.actions[index],
            self.threat_targets[index],
            self.attack_targets[index],
        )


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_spaces(dataset):
    obs_dim = dataset.obs.shape[-1]
    act_dim = dataset.actions.shape[-1]
    obs_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
    act_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(act_dim,), dtype=np.float32)
    return obs_space, act_space


def build_device(args):
    if args.cuda and torch.cuda.is_available():
        if args.cuda_device_id >= 0:
            return torch.device(f"cuda:{args.cuda_device_id}")
        return torch.device("cuda:0")
    return torch.device("cpu")


def prepare_batch(obs, actions, threat_targets, attack_targets, device):
    batch_size, num_agents = obs.shape[:2]
    obs = obs.reshape(batch_size * num_agents, -1).to(device=device, dtype=torch.float32)
    actions = actions.reshape(batch_size * num_agents, -1).to(device=device, dtype=torch.float32)
    threat_targets = threat_targets.to(device=device, dtype=torch.float32)
    attack_targets = attack_targets.to(device=device, dtype=torch.float32)
    return batch_size, obs, actions, threat_targets, attack_targets


def build_epoch_minibatch_subsets(dataset, mini_batches, epoch, seed):
    total_samples = len(dataset)
    if total_samples == 0:
        raise ValueError("Empty training dataset.")

    if mini_batches <= 1 or total_samples == 1:
        return [dataset]

    effective_batches = min(int(mini_batches), total_samples)
    rng = random.Random(seed + epoch * 9973)
    indices = list(range(total_samples))
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
        drop_last=False,
    )


def compute_r_value(sse, target_sum, target_square_sum, count):
    if count <= 0:
        return 0.0
    sst = target_square_sum - target_sum * target_sum / count
    if sst <= 1e-12:
        return 0.0
    return 1.0 - sse / sst


def evaluate(model, data_loader, device, threat_loss_weight, attack_loss_weight, return_stats=False):
    model.eval()
    total_loss = 0.0
    total_threat_loss = 0.0
    total_attack_loss = 0.0
    total_raw_threat_loss = 0.0
    total_raw_attack_loss = 0.0
    total_samples = 0
    threat_sse = 0.0
    attack_sse = 0.0
    threat_sum = 0.0
    attack_sum = 0.0
    threat_square_sum = 0.0
    attack_square_sum = 0.0
    threat_count = 0
    attack_count = 0

    with torch.no_grad():
        for obs, actions, threat_targets, attack_targets in data_loader:
            batch_size, obs, actions, threat_targets, attack_targets = prepare_batch(
                obs, actions, threat_targets, attack_targets, device
            )
            threat_pred, attack_pred, _ = model(obs, actions)

            threat_loss = F.mse_loss(threat_pred, threat_targets)
            attack_loss = F.mse_loss(attack_pred, attack_targets)
            loss = threat_loss_weight * threat_loss + attack_loss_weight * attack_loss

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

            total_loss += loss.item() * batch_size
            total_threat_loss += threat_loss_weight * threat_loss.item() * batch_size
            total_attack_loss += attack_loss_weight * attack_loss.item() * batch_size
            total_raw_threat_loss += threat_loss.item() * batch_size
            total_raw_attack_loss += attack_loss.item() * batch_size
            total_samples += batch_size

    metrics = {
        "loss": total_loss / max(total_samples, 1),
        "threat_loss": total_threat_loss / max(total_samples, 1),
        "attack_loss": total_attack_loss / max(total_samples, 1),
        "raw_threat_loss": total_raw_threat_loss / max(total_samples, 1),
        "raw_attack_loss": total_raw_attack_loss / max(total_samples, 1),
        "threat_r": compute_r_value(threat_sse, threat_sum, threat_square_sum, threat_count),
        "attack_r": compute_r_value(attack_sse, attack_sum, attack_square_sum, attack_count),
    }
    if return_stats:
        metrics.update(
            {
                "samples": int(total_samples),
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
            "raw_threat_loss": 0.0,
            "raw_attack_loss": 0.0,
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
        "samples": int(total_samples),
        "mini_batch_count": len(metrics_list),
        "threat_r": compute_r_value(threat_sse, threat_sum, threat_square_sum, threat_count),
        "attack_r": compute_r_value(attack_sse, attack_sum, attack_square_sum, attack_count),
    }
    for key in ("loss", "threat_loss", "attack_loss", "raw_threat_loss", "raw_attack_loss"):
        merged[key] = sum(item[key] * item["samples"] for item in metrics_list) / total_samples
    return merged


def evaluate_dataset(
    model,
    dataset,
    batch_size,
    mini_batches,
    epoch,
    seed,
    device,
    threat_loss_weight,
    attack_loss_weight,
    num_workers,
):
    subsets = build_epoch_minibatch_subsets(dataset, mini_batches, epoch, seed)
    metrics_list = []
    for subset in subsets:
        data_loader = DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=(device.type == "cuda"),
            drop_last=False,
        )
        metrics_list.append(
            evaluate(
                model=model,
                data_loader=data_loader,
                device=device,
                threat_loss_weight=threat_loss_weight,
                attack_loss_weight=attack_loss_weight,
                return_stats=True,
            )
        )
    return merge_eval_metrics(metrics_list)


def train_one_loader(
    model,
    optimizer,
    data_loader,
    device,
    max_grad_norm,
    threat_loss_weight,
    attack_loss_weight,
):
    model.train()
    total_loss = 0.0
    total_threat_loss = 0.0
    total_attack_loss = 0.0
    total_raw_threat_loss = 0.0
    total_raw_attack_loss = 0.0
    total_samples = 0

    for obs, actions, threat_targets, attack_targets in data_loader:
        batch_size, obs, actions, threat_targets, attack_targets = prepare_batch(
            obs, actions, threat_targets, attack_targets, device
        )
        threat_pred, attack_pred, _ = model(obs, actions)

        threat_loss = F.mse_loss(threat_pred, threat_targets)
        attack_loss = F.mse_loss(attack_pred, attack_targets)
        loss = threat_loss_weight * threat_loss + attack_loss_weight * attack_loss

        optimizer.zero_grad()
        loss.backward()
        if max_grad_norm is not None and max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        total_loss += loss.item() * batch_size
        total_threat_loss += threat_loss_weight * threat_loss.item() * batch_size
        total_attack_loss += attack_loss_weight * attack_loss.item() * batch_size
        total_raw_threat_loss += threat_loss.item() * batch_size
        total_raw_attack_loss += attack_loss.item() * batch_size
        total_samples += batch_size

    return {
        "loss": total_loss / max(total_samples, 1),
        "threat_loss": total_threat_loss / max(total_samples, 1),
        "attack_loss": total_attack_loss / max(total_samples, 1),
        "raw_threat_loss": total_raw_threat_loss / max(total_samples, 1),
        "raw_attack_loss": total_raw_attack_loss / max(total_samples, 1),
        "samples": int(total_samples),
    }


def merge_train_metrics(metrics_list):
    total_samples = sum(item["samples"] for item in metrics_list)
    if total_samples <= 0:
        return {
            "loss": 0.0,
            "threat_loss": 0.0,
            "attack_loss": 0.0,
            "raw_threat_loss": 0.0,
            "raw_attack_loss": 0.0,
            "samples": 0,
        }

    merged = {"samples": int(total_samples)}
    for key in ("loss", "threat_loss", "attack_loss", "raw_threat_loss", "raw_attack_loss"):
        merged[key] = sum(item[key] * item["samples"] for item in metrics_list) / total_samples
    return merged


def train_one_epoch(
    model,
    optimizer,
    dataset,
    batch_size,
    mini_batches,
    epoch,
    seed,
    device,
    max_grad_norm,
    threat_loss_weight,
    attack_loss_weight,
    num_workers,
):
    subsets = build_epoch_minibatch_subsets(dataset, mini_batches, epoch, seed)
    metrics_list = []
    for subset in subsets:
        data_loader = DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=(device.type == "cuda"),
            drop_last=False,
        )
        metrics_list.append(
            train_one_loader(
                model=model,
                optimizer=optimizer,
                data_loader=data_loader,
                device=device,
                max_grad_norm=max_grad_norm,
                threat_loss_weight=threat_loss_weight,
                attack_loss_weight=attack_loss_weight,
            )
        )

    metrics = merge_train_metrics(metrics_list)
    metrics["mini_batch_count"] = len(subsets)
    return metrics


def format_round_log(epoch, epochs, mini_batch_index, mini_batch_count, global_round, total_rounds, val_metrics, train_metrics, elapsed):
    return (
        f"\n"
        f"[Epoch {epoch:03d}/{epochs:03d} | Round {global_round:04d}/{total_rounds:04d} | "
        f"Mini-batch {mini_batch_index:02d}/{mini_batch_count:02d}]\n"
        f"  Val-before-train : loss={val_metrics['loss']:.6f} | "
        f"threat={val_metrics['threat_loss']:.6f} | attack={val_metrics['attack_loss']:.6f} | "
        f"threat_R={val_metrics['threat_r']:.4f} | attack_R={val_metrics['attack_r']:.4f} | "
        f"samples={val_metrics['samples']}\n"
        f"  Train            : loss={train_metrics['loss']:.6f} | "
        f"threat={train_metrics['threat_loss']:.6f} | attack={train_metrics['attack_loss']:.6f} | "
        f"samples={train_metrics['samples']}\n"
        f"  Time             : {elapsed:.2f}s"
    )


def save_checkpoint(path, model, optimizer, epoch, best_val_loss, args, train_dataset):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "best_val_loss": best_val_loss,
            "args": vars(args),
            "obs_shape": train_dataset.obs.shape,
            "actions_shape": train_dataset.actions.shape,
        },
        path,
    )


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


def build_run_dir(args):
    save_root = resolve_project_path(args.save_root)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_name = f"run-{timestamp}-seed{args.seed}"
    return save_root / args.experiment_name / run_name


def dump_config(path, args, dataset_paths, datasets):
    payload = {
        "args": vars(args),
        "dataset_paths": {key: normalize_path(value) for key, value in dataset_paths.items()},
        "dataset_shapes": {
            "train_obs": list(datasets["train"].obs.shape),
            "train_actions": list(datasets["train"].actions.shape),
            "val_obs": list(datasets["val"].obs.shape),
            "test_obs": list(datasets["test"].obs.shape),
        },
        "dataset_sample_counts": {
            "train_original": datasets["train"].original_sample_count,
            "train_kept": datasets["train"].kept_sample_count,
            "val_original": datasets["val"].original_sample_count,
            "val_kept": datasets["val"].kept_sample_count,
            "test_original": datasets["test"].original_sample_count,
            "test_kept": datasets["test"].kept_sample_count,
        },
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def get_parser():
    parser = argparse.ArgumentParser(description="Train AeroTAF on processed stage-1 datasets.")
    parser.add_argument("--dataset-dir", type=str, required=True, help="Processed dataset directory.")
    parser.add_argument("--train-file", type=str, default="train.npz", help="Training split filename.")
    parser.add_argument("--val-file", type=str, default="val.npz", help="Validation split filename.")
    parser.add_argument("--test-file", type=str, default="test.npz", help="Test split filename.")
    parser.add_argument("--experiment-name", type=str, default="AeroTAF-Stage1-K20-Baseline", help="Experiment name.")
    parser.add_argument("--save-root", type=str, default="scripts/results/AeroTAF", help="Checkpoint root directory.")
    parser.add_argument("--seed", type=int, default=1, help="Random seed.")
    parser.add_argument("--cuda", action="store_true", default=False, help="Use CUDA if available.")
    parser.add_argument("--cuda-device-id", type=int, default=0, help="CUDA device id.")
    parser.add_argument("--n-training-threads", type=int, default=1, help="Torch training thread count.")
    parser.add_argument("--batch-size", type=int, default=256, help="Mini-batch size.")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs.")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Adam weight decay.")
    parser.add_argument("--max-grad-norm", type=float, default=2.0, help="Gradient clipping max norm.")
    parser.add_argument(
        "--mini-batches",
        type=int,
        default=1,
        help="Split the training set into this many subsets and train all subsets once in each epoch.",
    )
    parser.add_argument("--threat-loss-weight", type=float, default=1.0, help="Weight for threat regression loss.")
    parser.add_argument("--attack-loss-weight", type=float, default=1.0, help="Weight for attack regression loss.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker count.")
    parser.add_argument("--save-interval", type=int, default=10, help="Latest checkpoint save interval.")
    parser.add_argument("--log-interval", type=int, default=1, help="Epoch logging interval.")
    parser.add_argument(
        "--sample-stride",
        type=int,
        default=1,
        help="Keep only timesteps whose within-episode index is a multiple of this stride. 1 keeps all samples.",
    )
    parser.add_argument("--num-agents", type=int, default=4, help="Number of ego agents.")
    parser.add_argument("--activation-id", type=int, default=1, help="0:Tanh, 1:ReLU, 2:LeakyReLU, 3:ELU.")
    parser.add_argument("--use-feature-normalization", action="store_true", default=False, help="Use LayerNorm on obs/action inputs.")
    parser.add_argument("--KQ-hidden-size", type=str, default="64 64", help="K/Q MLP hidden sizes.")
    parser.add_argument("--V-hidden-size", type=str, default="64 64", help="V MLP hidden sizes.")
    parser.add_argument("--AeroTAF-out-hidden-size", type=str, default="64", help="Threat/attack head hidden sizes.")
    parser.add_argument("--num-heads", type=int, default=4, help="Attention head count.")
    return parser


def main(args):
    parser = get_parser()
    all_args = parser.parse_args(args)

    dataset_dir = resolve_project_path(all_args.dataset_dir)
    dataset_paths = {
        "dataset_dir": dataset_dir,
        "train": dataset_dir / all_args.train_file,
        "val": dataset_dir / all_args.val_file,
        "test": dataset_dir / all_args.test_file,
    }
    for key in ("train", "val", "test"):
        if not dataset_paths[key].exists():
            raise FileNotFoundError(f"{key} split not found: {dataset_paths[key]}")

    set_seed(all_args.seed)
    torch.set_num_threads(all_args.n_training_threads)
    device = build_device(all_args)

    train_dataset = AeroTAFDataset(dataset_paths["train"], sample_stride=all_args.sample_stride)
    val_dataset = AeroTAFDataset(dataset_paths["val"], sample_stride=all_args.sample_stride)
    test_dataset = AeroTAFDataset(dataset_paths["test"], sample_stride=all_args.sample_stride)

    obs_space, act_space = build_spaces(train_dataset)
    model_args = AeroTAFArgs(all_args)
    model = PPOAeroTAF(model_args, obs_space, act_space, device=device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=all_args.lr,
        weight_decay=all_args.weight_decay,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=all_args.batch_size,
        shuffle=False,
        num_workers=all_args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    run_dir = build_run_dir(all_args)
    run_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = run_dir / "AeroTAF_best.pt"
    latest_ckpt_path = run_dir / "AeroTAF_latest.pt"
    log_path = run_dir / "train_log.csv"
    config_path = run_dir / "config.json"
    test_metrics_path = run_dir / "test_metrics.json"

    dump_config(
        config_path,
        all_args,
        dataset_paths,
        {"train": train_dataset, "val": val_dataset, "test": test_dataset},
    )

    logging.info(f"device: {device}")
    logging.info(f"dataset dir: {normalize_path(dataset_dir)}")
    logging.info(f"train split: {normalize_path(dataset_paths['train'])}")
    logging.info(f"val split: {normalize_path(dataset_paths['val'])}")
    logging.info(f"test split: {normalize_path(dataset_paths['test'])}")
    logging.info(f"run dir: {normalize_path(run_dir)}")
    logging.info(f"train samples: {len(train_dataset)}")
    logging.info(f"val samples: {len(val_dataset)}")
    logging.info(f"test samples: {len(test_dataset)}")
    logging.info(
        f"sample stride: {all_args.sample_stride} | "
        f"train kept/original={train_dataset.kept_sample_count}/{train_dataset.original_sample_count} | "
        f"val kept/original={val_dataset.kept_sample_count}/{val_dataset.original_sample_count} | "
        f"test kept/original={test_dataset.kept_sample_count}/{test_dataset.original_sample_count}"
    )
    logging.info(f"train obs shape: {train_dataset.obs.shape}")
    logging.info(f"train actions shape: {train_dataset.actions.shape}")
    logging.info(
        f"loss weights: threat={all_args.threat_loss_weight}, attack={all_args.attack_loss_weight}"
    )
    effective_mini_batches = max(1, min(int(all_args.mini_batches), len(train_dataset), len(val_dataset)))
    total_rounds = all_args.epochs * effective_mini_batches
    logging.info(
        f"mini-batches per epoch: {effective_mini_batches} | total paired rounds: {total_rounds}"
    )
    logging.info("round order: validate one val subset -> train one train subset")

    best_val_loss = float("inf")

    for epoch in range(1, all_args.epochs + 1):
        train_subsets = build_epoch_minibatch_subsets(train_dataset, effective_mini_batches, epoch, all_args.seed)
        val_subsets = build_epoch_minibatch_subsets(val_dataset, effective_mini_batches, epoch, all_args.seed + 100000)

        for mini_batch_index, (val_subset, train_subset) in enumerate(zip(val_subsets, train_subsets), start=1):
            round_start_time = time.time()
            global_round = (epoch - 1) * effective_mini_batches + mini_batch_index

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
                return_stats=True,
            )

            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                save_checkpoint(best_ckpt_path, model, optimizer, global_round - 1, best_val_loss, all_args, train_dataset)
                logging.info(f"best checkpoint updated at round {global_round}: {normalize_path(best_ckpt_path)}")

            train_loader = build_loader(
                train_subset,
                batch_size=all_args.batch_size,
                shuffle=True,
                num_workers=all_args.num_workers,
                device=device,
            )
            train_metrics = train_one_loader(
                model=model,
                optimizer=optimizer,
                data_loader=train_loader,
                device=device,
                max_grad_norm=all_args.max_grad_norm,
                threat_loss_weight=all_args.threat_loss_weight,
                attack_loss_weight=all_args.attack_loss_weight,
            )
            elapsed = time.time() - round_start_time

            row = {
                "epoch": epoch,
                "mini_batch_index": mini_batch_index,
                "mini_batch_total": effective_mini_batches,
                "global_round": global_round,
                "total_rounds": total_rounds,
                "val_before_train_loss": f"{val_metrics['loss']:.8f}",
                "val_before_train_threat_loss": f"{val_metrics['threat_loss']:.8f}",
                "val_before_train_attack_loss": f"{val_metrics['attack_loss']:.8f}",
                "val_before_train_raw_threat_loss": f"{val_metrics['raw_threat_loss']:.8f}",
                "val_before_train_raw_attack_loss": f"{val_metrics['raw_attack_loss']:.8f}",
                "val_before_train_threat_r": f"{val_metrics['threat_r']:.8f}",
                "val_before_train_attack_r": f"{val_metrics['attack_r']:.8f}",
                "val_samples": val_metrics["samples"],
                "train_loss": f"{train_metrics['loss']:.8f}",
                "train_threat_loss": f"{train_metrics['threat_loss']:.8f}",
                "train_attack_loss": f"{train_metrics['attack_loss']:.8f}",
                "train_raw_threat_loss": f"{train_metrics['raw_threat_loss']:.8f}",
                "train_raw_attack_loss": f"{train_metrics['raw_attack_loss']:.8f}",
                "train_samples": train_metrics["samples"],
                "time_sec": f"{elapsed:.4f}",
            }
            append_csv_row(log_path, row, write_header=(global_round == 1))

            if epoch % all_args.log_interval == 0 or epoch == 1 or epoch == all_args.epochs:
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
                    )
                )

        if epoch % all_args.save_interval == 0 or epoch == all_args.epochs:
            save_checkpoint(latest_ckpt_path, model, optimizer, epoch, best_val_loss, all_args, train_dataset)
            logging.info(f"latest checkpoint saved: {normalize_path(latest_ckpt_path)}")

    final_val_metrics = evaluate_dataset(
        model=model,
        dataset=val_dataset,
        batch_size=all_args.batch_size,
        mini_batches=all_args.mini_batches,
        epoch=all_args.epochs + 1,
        seed=all_args.seed + 100000,
        device=device,
        threat_loss_weight=all_args.threat_loss_weight,
        attack_loss_weight=all_args.attack_loss_weight,
        num_workers=all_args.num_workers,
    )
    if final_val_metrics["loss"] < best_val_loss:
        best_val_loss = final_val_metrics["loss"]
        save_checkpoint(best_ckpt_path, model, optimizer, all_args.epochs, best_val_loss, all_args, train_dataset)
        logging.info(f"best checkpoint updated after final epoch: {normalize_path(best_ckpt_path)}")

    logging.info(
        f"final val loss={final_val_metrics['loss']:.6f} "
        f"(threat={final_val_metrics['threat_loss']:.6f}, attack={final_val_metrics['attack_loss']:.6f}, "
        f"threat_R={final_val_metrics['threat_r']:.4f}, attack_R={final_val_metrics['attack_r']:.4f})"
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
    )
    with open(test_metrics_path, "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2, ensure_ascii=False)

    logging.info(
        f"test loss={test_metrics['loss']:.6f} "
        f"(threat={test_metrics['threat_loss']:.6f}, attack={test_metrics['attack_loss']:.6f}, "
        f"threat_R={test_metrics['threat_r']:.4f}, attack_R={test_metrics['attack_r']:.4f})"
    )
    logging.info(f"test metrics saved: {normalize_path(test_metrics_path)}")
    logging.info("Done.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    default_args = [
        "--dataset-dir", "datasets/aerotaf/4v4_shoot_mappo_pool_stage1/processed_stage1_K20_field_temporal",
        "--experiment-name", "AeroTAF-Stage1-K20-Baseline",
        "--save-root", "scripts/results/AeroTAF",
        "--seed", "1",
        "--n-training-threads", "1",
        "--batch-size", "256",
        "--epochs", "10",
        "--mini-batches", "10",
        "--lr", "3e-5",
        "--weight-decay", "1e-4",
        "--max-grad-norm", "2.0",
        "--threat-loss-weight", "1.0",
        "--attack-loss-weight", "8.0",
        "--num-workers", "0",
        "--save-interval", "10",
        "--log-interval", "1",
        "--sample-stride", "1",
        "--num-agents", "4",
        "--activation-id", "1",
        "--KQ-hidden-size", "128 128",
        "--V-hidden-size", "128 128",
        "--AeroTAF-out-hidden-size", "128",
        "--num-heads", "4",
        "--use-feature-normalization",
        # "--cuda",
        # "--cuda-device-id", "0",
        # "--train-file", "train.npz",
        # "--val-file", "val.npz",
        # "--test-file", "test.npz",
    ]
    main(sys.argv[1:] if len(sys.argv) > 1 else default_args)
