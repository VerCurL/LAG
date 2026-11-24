"""用于 smoke 测试和契约验证的简单航迹跟踪策略。"""

from laglayer3.domain.models import ExecutionAction, LocalObservation, RoutePrediction


class FixedRouteExecutionProvider:
    """根据首个航迹点生成基础的高度、航向和速度调整。"""
    def reset(self, aircraft_id: str) -> None:
        """清理指定飞机的执行状态。"""
        return None

    def act(self, observation: LocalObservation, route: RoutePrediction) -> ExecutionAction:
        """根据当前飞机状态和航迹首点生成离散执行动作。"""
        aircraft = observation.aircraft
        if not route.points:
            return ExecutionAction(1, 2, 1)
        forward, right, altitude, speed = route.points[0]
        altitude_action = 0 if aircraft.altitude_m < altitude - 50 else 2 if aircraft.altitude_m > altitude + 50 else 1
        speed_action = 0 if aircraft.speed_mps < speed - 10 else 2 if aircraft.speed_mps > speed + 10 else 1
        heading_action = 1 if abs(right) < 100 else (3 if right > 0 else 1)
        return ExecutionAction(altitude_action, heading_action, speed_action)
