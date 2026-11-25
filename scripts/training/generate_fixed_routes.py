
"""根据执行训练配置预生成并保存固定航迹。"""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from laglayer3.config import load_yaml_config
from laglayer3.planning.route_library import build_route_cache


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/training/execution/fixed_routes_ppo.yaml")
    args = parser.parse_args()
    route = load_yaml_config(args.config)["route"]
    print(build_route_cache(ROOT / route["spec"], ROOT / route["cache"]))


if __name__ == "__main__":
    main()
