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

from algorithms.utils.AeroTAF import AeroTAFBase
from scripts.AeroTAF.collector.path_utils import normalize_path, resolve_project_path


class AeroTAFArgs:
    def __init__(self, args):
        self.num_agents = args.num_agents
        self.activation_id = args.activation_id
        self.use_feature_normalization = args.use_feature_normalization
        self.num_heads = args.num_heads
        self.KQ_hidden_size = args.KQ_hidden_size
        self.V_hidden_size = args.V_hidden_size
        self.output_hidden_size = args.output_hidden_size


class PPOAeroTAF(nn.Module):
    def __init__(self, args, obs_space, act_space, device=torch.device("cpu")):
        super().__init__()
        self.num_agents = args.num_agents
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.AeroTAF = AeroTAFBase(
            obs_space=obs_space,
            act_space=act_space,
            agent_num=args.num_agents,
            head_num=args.num_heads,
            KQ_hidden_size=args.KQ_hidden_size,
            V_hidden_size=args.V_hidden_size,
            output_hidden_size=args.output_hidden_size,
            activation_id=args.activation_id,
            use_feature_normalization=args.use_feature_normalization,
        )
        self.to(device)

    def forward(self, obs, actions):
        obs = torch.as_tensor(obs, **self.tpdv)
        actions = torch.as_tensor(actions, **self.tpdv)
        return self.AeroTAF(obs, actions)


class AeroTAFStepDataset(Dataset):
    def __init__(self, npz_path):
        self.path = str(npz_path)
        with np.load(npz_path, allow_pickle=True) as data:
            required = ["obs", "actions", "threat_targets", "attack_targets"]
            missing = [key for key in required if key not in data.files]
            if missing:
                raise KeyError(f"{npz_path} missing keys: {missing}")

            self.obs = data["obs"].astype(np.float32, copy=False)
            self.actions = data["actions"].astype(np.float32, copy=False)
            self.threat_targets = data["threat_targets"].astype(np.float32, copy=False)
            self.attack_targets = data["attack_targets"].astype(np.float32, copy=False)

        if self.obs.ndim != 4:
            raise ValueError(f"{npz_path}: obs must be [windows, time, agents, obs_dim], got {self.obs.shape}")
        if self.actions.ndim != 4:
            raise ValueError(f"{npz_path}: actions must be [windows, time, agents, act_dim], got {self.actions.shape}")
        if self.threat_targets.ndim != 3:
            raise ValueError(f"{npz_path}: threat_targets must be [windows, time, 1], got {self.threat_targets.shape}")
        if self.attack_targets.ndim != 3:
            raise ValueError(f"{npz_path}: attack_targets must be [windows, time, 1], got {self.attack_targets.shape}")
        if self.actions.shape[:3] != self.obs.shape[:3]:
            raise ValueError(f"{npz_path}: actions prefix {self.actions.shape[:3]} != obs prefix {self.obs.shape[:3]}")
        if self.threat_targets.shape[:2] != self.obs.shape[:2]:
            raise ValueError(f"{npz_path}: threat prefix {self.threat_targets.shape[:2]} != obs prefix {self.obs.shape[:2]}")
        if self.attack_targets.shape[:2] != self.obs.shape[:2]:
            raise ValueError(f"{npz_path}: attack prefix {self.attack_targets.shape[:2]} != obs prefix {self.obs.shape[:2]}")

        self.num_windows = int(self.obs.shape[0])
        self.window_length = int(self.obs.shape[1])
        self.num_agents = int(self.obs.shape[2])
        self.num_steps = self.num_windows * self.window_length

    def __len__(self):
        return self.num_steps

    def __getitem__(self, index):
        window_index = int(index) // self.window_length
        time_index = int(index) % self.window_length
        return (
            self.obs[window_index, time_index],
            self.actions[window_index, time_index],
            self.threat_targets[window_index, time_index],
            self.attack_targets[window_index, time_index],
        )


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
    obs_dim = int(dataset.obs.shape[-1])
    act_dim = int(dataset.actions.shape[-1])
    obs_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
    act_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(act_dim,), dtype=np.float32)
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
    obs, actions, threat_targets, attack_targets = batch
    obs = torch.as_tensor(obs, dtype=torch.float32, device=device)
    actions = torch.as_tensor(actions, dtype=torch.float32, device=device)
    threat_targets = torch.as_tensor(threat_targets, dtype=torch.float32, device=device).reshape(obs.shape[0], -1)
    attack_targets = torch.as_tensor(attack_targets, dtype=torch.float32, device=device).reshape(obs.shape[0], -1)

    if obs.ndim != 3:
        raise ValueError(f"Expected obs [B,N,D], got {tuple(obs.shape)}")
    batch_size, num_agents = obs.shape[:2]
    obs = obs.reshape(batch_size * num_agents, -1)
    actions = actions.reshape(batch_size * num_agents, -1)
    return obs, actions, threat_targets, attack_targets, int(batch_size)


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


def update_totals(totals, loss, threat_loss, attack_loss, threat_pred, attack_pred, threat_targets, attack_targets, count, args):
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
    obs, actions, threat_targets, attack_targets, count = prepare_batch(batch, device)
    threat_pred, attack_pred = model(obs, actions)
    raw_threat_loss = F.mse_loss(threat_pred, threat_targets)
    raw_attack_loss = F.mse_loss(attack_pred, attack_targets)
    loss = args.threat_loss_weight * raw_threat_loss + args.attack_loss_weight * raw_attack_loss
    return loss, raw_threat_loss, raw_attack_loss, threat_pred, attack_pred, threat_targets, attack_targets, count


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
            f"| steps={count} | {elapsed:.2f}s"
        )
        update_totals(totals, loss, raw_threat_loss, raw_attack_loss, threat_pred, attack_pred, threat_targets, attack_targets, count, args)

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
            update_totals(totals, loss, raw_threat_loss, raw_attack_loss, threat_pred, attack_pred, threat_targets, attack_targets, count, args)
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
        "dataset_summary": {
            key: {
                "windows": int(dataset.num_windows),
                "window_length": int(dataset.window_length),
                "time_step_samples": int(len(dataset)),
                "obs_shape": list(dataset.obs.shape),
                "actions_shape": list(dataset.actions.shape),
            }
            for key, dataset in datasets.items()
        },
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def get_parser():
    parser = argparse.ArgumentParser(description="Train AeroTAF on window datasets by shuffling individual timesteps.")
    parser.add_argument("--dataset-dir", type=str, required=True, help="Processed dataset directory.")
    parser.add_argument("--train-file", type=str, default="train.npz", help="Training split filename.")
    parser.add_argument("--val-file", type=str, default="val.npz", help="Validation split filename.")
    parser.add_argument("--test-file", type=str, default="test.npz", help="Test split filename.")
    parser.add_argument("--experiment-name", type=str, default="AeroTAF-Step-K20-L50-S30", help="Experiment name.")
    parser.add_argument("--save-root", type=str, default="scripts/results/AeroTAF", help="Checkpoint root directory.")
    parser.add_argument("--seed", type=int, default=1, help="Random seed.")
    parser.add_argument("--cuda", action="store_true", default=False, help="Use CUDA if available.")
    parser.add_argument("--cuda-device-id", type=int, default=0, help="CUDA device id.")
    parser.add_argument("--n-training-threads", type=int, default=1, help="Torch training thread count.")
    parser.add_argument("--mini-epoch", type=int, default=100, help="Number of parameter updates per epoch; each update uses roughly train_steps / mini_epoch shuffled timesteps.")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs.")
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
    parser.add_argument("--output-hidden-size", type=str, default="64 32", help="Threat/attack output head hidden sizes.")
    parser.add_argument("--num-heads", type=int, default=4, help="Spatial attention head count.")
    return parser


def main(args):
    parser = get_parser()
    all_args = parser.parse_args(args)

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
        "train": AeroTAFStepDataset(dataset_paths["train"]),
        "val": AeroTAFStepDataset(dataset_paths["val"]),
        "test": AeroTAFStepDataset(dataset_paths["test"]),
    }
    all_args.effective_mini_epoch = max(1, min(int(all_args.mini_epoch), len(datasets["train"])))
    all_args.eval_batch_size = max(1, int(np.ceil(len(datasets["train"]) / all_args.effective_mini_epoch)))

    obs_space, act_space = build_spaces(datasets["train"])
    model = PPOAeroTAF(AeroTAFArgs(all_args), obs_space, act_space, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=all_args.lr, weight_decay=all_args.weight_decay)
    loaders = {
        "val": build_eval_loader(datasets["val"], all_args, device),
        "test": build_eval_loader(datasets["test"], all_args, device),
    }

    run_dir = build_run_dir(all_args)
    run_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = run_dir / "AeroTAF_best.pt"
    latest_ckpt_path = run_dir / "AeroTAF_latest.pt"
    epoch_log_path = run_dir / "epoch_log.csv"
    config_path = run_dir / "config.json"
    test_metrics_path = run_dir / "test_metrics.json"
    dump_config(config_path, all_args, dataset_paths, datasets)

    logging.info("=" * 72)
    logging.info("AeroTAF Step Training")
    logging.info("=" * 72)
    logging.info(f"device     : {device}")
    logging.info(f"dataset    : {normalize_path(dataset_dir)}")
    logging.info(f"run dir    : {normalize_path(run_dir)}")
    logging.info(
        f"samples    : train={len(datasets['train'])} val={len(datasets['val'])} test={len(datasets['test'])} "
        f"| mini_epoch={all_args.effective_mini_epoch} | steps/update~{all_args.eval_batch_size}"
    )
    logging.info(
        f"windows    : train={datasets['train'].num_windows} val={datasets['val'].num_windows} test={datasets['test'].num_windows} "
        f"| length={datasets['train'].window_length}"
    )
    logging.info(
        f"loss w     : threat={all_args.threat_loss_weight} attack={all_args.attack_loss_weight} | lr={all_args.lr:.2e}"
    )
    logging.info("-" * 72)

    best_val_loss = float("inf")
    initial_val = evaluate(model, loaders["val"], device, all_args)
    best_val_loss = initial_val["loss"]
    save_checkpoint(best_ckpt_path, model, optimizer, 0, best_val_loss, all_args)
    logging.info(
        f"[INIT] val={initial_val['loss']:.6f} "
        f"| raw(th/a)=({initial_val['threat_loss']:.6f}/{initial_val['attack_loss']:.6f}) "
        f"| R(th/a)=({initial_val['threat_r']:.4f}/{initial_val['attack_r']:.4f})"
    )

    for epoch in range(1, all_args.epochs + 1):
        start_time = time.time()
        train_loader = build_train_loader(datasets["train"], all_args, device, epoch)
        train_metrics = train_one_epoch(model, train_loader, device, all_args, optimizer)
        val_metrics = evaluate(model, loaders["val"], device, all_args)
        elapsed = time.time() - start_time

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            save_checkpoint(best_ckpt_path, model, optimizer, epoch, best_val_loss, all_args)
            logging.info(f"[BEST] epoch={epoch:03d} val={best_val_loss:.6f} -> {normalize_path(best_ckpt_path)}")

        if epoch % all_args.save_interval == 0 or epoch == all_args.epochs:
            save_checkpoint(latest_ckpt_path, model, optimizer, epoch, best_val_loss, all_args)

        append_csv_row(
            epoch_log_path,
            {
                "epoch": epoch,
                "train_loss": f"{train_metrics['loss']:.8f}",
                "train_raw_threat_loss": f"{train_metrics['threat_loss']:.8f}",
                "train_raw_attack_loss": f"{train_metrics['attack_loss']:.8f}",
                "train_updates": int(train_metrics["updates"]),
                "val_loss": f"{val_metrics['loss']:.8f}",
                "val_raw_threat_loss": f"{val_metrics['threat_loss']:.8f}",
                "val_raw_attack_loss": f"{val_metrics['attack_loss']:.8f}",
                "val_threat_r": f"{val_metrics['threat_r']:.8f}",
                "val_attack_r": f"{val_metrics['attack_r']:.8f}",
                "best_val_loss": f"{best_val_loss:.8f}",
                "time_sec": f"{elapsed:.4f}",
            },
            write_header=(epoch == 1),
        )

        if epoch % all_args.log_interval == 0 or epoch == 1 or epoch == all_args.epochs:
            logging.info(
                f"[E{epoch:03d}/{all_args.epochs:03d}] "
                f"train={train_metrics['loss']:.5f} "
                f"| val={val_metrics['loss']:.5f} "
                f"| raw_val(th/a)=({val_metrics['threat_loss']:.5f}/{val_metrics['attack_loss']:.5f}) "
                f"| R(th/a)=({val_metrics['threat_r']:.3f}/{val_metrics['attack_r']:.3f}) "
                f"| updates={train_metrics['updates']} "
                f"| {elapsed:.1f}s"
            )

    if best_ckpt_path.exists():
        checkpoint = torch.load(best_ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        logging.info(f"loaded best checkpoint for final test: {normalize_path(best_ckpt_path)}")

    test_metrics = evaluate(model, loaders["test"], device, all_args)
    with open(test_metrics_path, "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2, ensure_ascii=False)

    logging.info("-" * 72)
    logging.info(
        f"[TEST] loss={test_metrics['loss']:.6f} "
        f"| raw(th/a)=({test_metrics['threat_loss']:.6f}/{test_metrics['attack_loss']:.6f}) "
        f"| R(th/a)=({test_metrics['threat_r']:.4f}/{test_metrics['attack_r']:.4f})"
    )
    logging.info(f"test metrics saved: {normalize_path(test_metrics_path)}")
    logging.info("Done.")


if __name__ == "__main__":
    default_args = [
        "--dataset-dir", "datasets/aerotaf/4v4_shoot_mappo_pool_stage1/processed_stage1_K20_L50_S30",
        "--experiment-name", "AeroTAF-Stage1-K20-L50-S30",
        "--save-root", "scripts/results/AeroTAF",
        "--seed", "1",
        "--n-training-threads", "1",
        "--mini-epoch", "10",
        "--epochs", "10",
        "--lr", "3e-5",
        "--weight-decay", "1e-4",
        "--max-grad-norm", "2.0",
        "--threat-loss-weight", "1.0",
        "--attack-loss-weight", "8.0",
        "--num-workers", "0",
        "--save-interval", "10",
        "--log-interval", "1",
        "--num-agents", "4",
        "--activation-id", "1",
        "--KQ-hidden-size", "128 128",
        "--V-hidden-size", "128 128",
        "--output-hidden-size", "64 32",
        "--num-heads", "4",
        "--use-feature-normalization",
        # "--cuda",
        # "--cuda-device-id", "0",
    ]
    main(sys.argv[1:] if len(sys.argv) > 1 else default_args)
