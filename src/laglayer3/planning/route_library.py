
"""预生成和加载版本化的固定航迹点列。"""

import hashlib
import json
import math
from pathlib import Path

from laglayer3.config import load_yaml_config
from .routes import make_route

SCHEMA = "laglayer3.fixed_routes.v1"


def _read_spec(path):
    spec = load_yaml_config(path)
    if spec["schema_version"] != "laglayer3.fixed_route_spec.v1":
        raise ValueError("航迹定义版本不匹配")
    fingerprint = hashlib.sha256(
        json.dumps(spec, sort_keys=True, allow_nan=False).encode("utf-8")
    ).hexdigest()
    return spec, fingerprint


def build_route_cache(spec_path, output_path):
    spec, fingerprint = _read_spec(spec_path)
    interval = float(spec["interval_seconds"])
    horizon = float(spec["horizon_seconds"])
    if not 0 < interval <= horizon or not math.isclose(horizon / interval, round(horizon / interval)):
        raise ValueError("航迹时长须为点间隔的正整数倍")
    speed = float(spec["initial_speed_mps"])
    altitude = float(spec["initial_altitude_m"])
    if speed <= 0 or altitude <= 0 or not spec["routes"]:
        raise ValueError("初始速度、高度须为正值，航迹集不能为空")
    routes = {}
    for name, settings in spec["routes"].items():
        yaw_rate = math.radians(float(settings["turn_rate_deg_s"]))
        climb_rate = float(settings["climb_rate_mps"])
        if abs(climb_rate) >= speed:
            raise ValueError("升降速率必须小于参考速度")
        horizontal_speed = math.sqrt(speed**2 - climb_rate**2)
        points = []
        for index in range(1, round(horizon / interval) + 1):
            t = index * interval
            if yaw_rate:
                forward = horizontal_speed * math.sin(yaw_rate * t) / yaw_rate
                right = horizontal_speed * (1 - math.cos(yaw_rate * t)) / yaw_rate
            else:
                forward, right = horizontal_speed * t, 0.0
            points.append([forward, right, altitude + climb_rate * t, speed])
        routes[name] = points
    data = dict(schema=SCHEMA, spec_sha256=fingerprint, interval_seconds=interval,
                initial_speed_mps=speed, initial_altitude_m=altitude, routes=routes)
    payload = json.dumps(data, indent=2, allow_nan=False)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload + "\n", encoding="utf-8")
    return path


class FixedRouteLibrary:
    def __init__(self, spec_path, cache_path):
        spec, fingerprint = _read_spec(spec_path)
        path = Path(cache_path)
        if not path.is_file():
            raise FileNotFoundError("缺少航迹缓存 {}，请先运行 generate_fixed_routes.py".format(path))
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("schema") != SCHEMA or data.get("spec_sha256") != fingerprint:
            raise ValueError("航迹缓存版本或配置不匹配，请重新生成")
        self.interval_seconds = float(spec["interval_seconds"])
        self.horizon_seconds = float(spec["horizon_seconds"])
        self.initial_speed_mps = float(spec["initial_speed_mps"])
        self.initial_altitude_m = float(spec["initial_altitude_m"])
        count = round(self.horizon_seconds / self.interval_seconds)
        self.routes = data["routes"]
        if self.routes.keys() != spec["routes"].keys() or any(
            len(points) != count or any(len(point) != 4 or not all(map(math.isfinite, point))
                                        for point in points)
            for points in self.routes.values()
        ):
            raise ValueError("航迹缓存中的名称、点数或数值无效")

    def bind(self, name, aircraft):
        return make_route(aircraft.aircraft_id, self.routes[name], self.interval_seconds,
                          aircraft.position_ned_m, aircraft.heading_deg, route_mode=name)
