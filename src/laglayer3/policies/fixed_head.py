"""用于本地闭环测试的确定性首脑策略。"""

from typing import Dict, Mapping

from laglayer3.domain.models import RoutePrediction, TeamObservation, WeaponIntent


class FixedRouteHeadProvider:
    """为每架飞机生成保持当前高度和速度的直线航迹。"""
    def __init__(self, horizon_seconds: float = 20.0, interval_seconds: float = 1.0) -> None:
        """设置航迹预测时域和相邻航迹点的时间间隔。"""
        if horizon_seconds <= 0 or interval_seconds <= 0:
            raise ValueError("route horizon and interval must be positive")
        self.horizon_seconds = float(horizon_seconds)
        self.interval_seconds = float(interval_seconds)
        self.reset()

    def reset(self) -> None:
        """清除上一次首脑决策的时间记录。"""
        self.last_timestamp = None

    def decide(self, observation: TeamObservation, timestamp: float) -> Mapping[str, object]:
        """根据当前飞机状态生成直线航迹和空武器意图。"""
        routes: Dict[str, RoutePrediction] = {}
        weapons: Dict[str, WeaponIntent] = {}
        count = max(1, int(round(self.horizon_seconds / self.interval_seconds)))
        for aircraft in observation.own_aircraft:
            points = []
            for index in range(1, count + 1):
                forward = aircraft.speed_mps * self.interval_seconds * index
                points.append((forward, 0.0, aircraft.altitude_m, aircraft.speed_mps))
            routes[aircraft.aircraft_id] = RoutePrediction(
                aircraft_id=aircraft.aircraft_id,
                points=tuple(points),
                interval_seconds=self.interval_seconds,
                origin_position_ned_m=aircraft.position_ned_m,
                origin_heading_deg=aircraft.heading_deg,
                confidence=1.0,
                route_mode="fixed_forward",
            )
            weapons[aircraft.aircraft_id] = WeaponIntent(aircraft_id=aircraft.aircraft_id)
        self.last_timestamp = float(timestamp)
        return {"routes": routes, "weapons": weapons, "timestamp": timestamp}
