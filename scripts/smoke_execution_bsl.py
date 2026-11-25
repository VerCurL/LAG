
"""检查固定航迹环境中的 BSL 调用次数和 JSBSim 时钟。"""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import math
import torch

from laglayer3.config import load_yaml_config
from laglayer3.learning.execution.env import make_execution_env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/training/execution/fixed_routes_ppo.yaml")
    args = parser.parse_args()
    torch.set_num_threads(1)
    env = make_execution_env(load_yaml_config(args.config), ROOT, seed=0)
    for route in ("straight", "turn_left"):
        env.reset(route)
        for _ in range(2):
            _, reward, _, _, info = env.step([1, 2, 1])
        snapshot = env.simulation.snapshot()
        controller = env.simulation.control_adapter.controller
        assert controller.control_steps[env.aircraft.aircraft_id] == 2
        assert math.isclose(snapshot.timestamp, 0.4, abs_tol=1e-9)
        print("route={} t={:.1f} reward={:.4f} position_error_m={:.2f}".format(
            route, snapshot.timestamp, reward, info["position_error_m"]
        ))


if __name__ == "__main__":
    main()
