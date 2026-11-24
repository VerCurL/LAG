from pathlib import Path
from typing import Any, Mapping

import yaml


def load_yaml_config(path: str) -> Mapping[str, Any]:
    """中文说明：读取并校验 YAML 配置文件的根对象。"""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError("configuration root must be a mapping")
    return value
