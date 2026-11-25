"""检查 LAGLayer3 配置是否包含第一版运行所需的顶层区段。"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from laglayer3.config import load_yaml_config


def main() -> None:
    """读取配置文件，检查必需区段并输出校验结果。"""
    parser = argparse.ArgumentParser(description="校验 LAGLayer3 YAML 配置")
    parser.add_argument("config", help="待校验的 YAML 配置路径")
    args = parser.parse_args()
    config = load_yaml_config(args.config)
    required = {"runtime", "route", "head", "execution", "flight_control", "simulation", "training"}
    missing = required.difference(config)
    if missing:
        # 缺少顶层区段时直接失败，避免运行过程中才暴露配置错误。
        raise SystemExit("missing config sections: {}".format(sorted(missing)))
    print("valid configuration: {}".format(args.config))


if __name__ == "__main__":
    main()
