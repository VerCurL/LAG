
"""用相同的飞控和仿真链逐条评估固定航迹。"""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import torch

from laglayer3.config import load_yaml_config
from laglayer3.learning.execution.env import make_execution_env
from laglayer3.learning.execution.ppo_trainer import evaluate_action, load_execution_model
from laglayer3.learning.execution.tracking import ERROR_KEYS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/training/execution/fixed_routes_ppo.yaml")
    parser.add_argument("--route", action="append", help="可重复指定，默认评估配置中的全部航迹")
    args = parser.parse_args()
    torch.set_num_threads(1)
    config = load_yaml_config(args.config)
    env = make_execution_env(config, ROOT, seed=0)
    model = load_execution_model(args.checkpoint, env, config["ppo"], "cpu")
    for name in args.route or env.route_names:
        observation = env.reset(name)
        hidden = None
        cumulative_reward = 0.0
        errors = dict.fromkeys(ERROR_KEYS, 0.0)
        while not env.done:
            action, _, _, hidden = evaluate_action(model, observation, hidden, "cpu", deterministic=True)
            observation, reward, _, _, info = env.step(action)
            cumulative_reward += reward
            for key in errors:
                errors[key] += info[key]
        report = dict(info, episode_return=cumulative_reward,
                      mean_errors={key: value / env.steps for key, value in errors.items()})
        print(json.dumps(report, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
