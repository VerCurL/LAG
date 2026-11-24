import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from laglayer3.config import load_yaml_config


def main() -> None:
    """中文说明：解析命令行参数并运行当前脚本的主流程。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()
    config = load_yaml_config(args.config)
    required = {"runtime", "route", "head", "execution", "flight_control", "simulation", "training"}
    missing = required.difference(config)
    if missing:
        raise SystemExit("missing config sections: {}".format(sorted(missing)))
    print("valid configuration: {}".format(args.config))


if __name__ == "__main__":
    main()
