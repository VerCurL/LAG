
"""用固定航迹和本地 bsl_aggressive 训练单机执行策略。"""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import torch

from laglayer3.config import load_yaml_config
from laglayer3.learning.execution.env import make_execution_env
from laglayer3.learning.execution.ppo_trainer import ExecutionPPO


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/training/execution/fixed_routes_ppo.yaml")
    parser.add_argument("--updates", type=int, default=100)
    parser.add_argument("--rollout-steps", type=int)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.updates < 1 or (args.rollout_steps is not None and args.rollout_steps < 1):
        parser.error("更新轮数与采样步数须为正整数")
    args.output.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output / "metrics.jsonl"
    checkpoint = args.output / "execution.pt"
    if metrics_path.exists() or checkpoint.exists():
        parser.error("请为新训练指定新的输出目录")
    torch.set_num_threads(1)
    config = load_yaml_config(args.config)
    if args.rollout_steps is not None:
        config["ppo"]["rollout_steps"] = args.rollout_steps
    env = make_execution_env(config, ROOT, args.seed)
    trainer = ExecutionPPO(env, config["ppo"], args.seed, args.device)
    (args.output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    with metrics_path.open("x", encoding="utf-8") as stream:
        for index in range(1, args.updates + 1):
            transitions, episodes = trainer.collect()
            record = dict(trainer.update(transitions), update=index, steps=trainer.total_steps, episodes=episodes)
            line = json.dumps(record, ensure_ascii=False, allow_nan=False)
            stream.write(line + "\n")
            stream.flush()
            print(line, flush=True)
            trainer.save(checkpoint)


if __name__ == "__main__":
    main()
