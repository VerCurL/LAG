
"""按仿真时间评价航迹拟合和飞行质量。"""

import math

import numpy as np

from laglayer3.planning.frames import ned_to_route_frame

METRICS = ("position", "heading", "altitude", "speed", "energy", "action_change")
ERROR_KEYS = ("position_error_m", "heading_error_deg", "altitude_error_m", "speed_error_mps")


def reference_at(points, interval, elapsed):
    """在含初始点的参考序列上插值，使用航迹切线作为参考航向。"""
    coordinate = min(max(elapsed / interval, 0.0), len(points) - 1)
    left = min(int(coordinate), len(points) - 2)
    delta = points[left + 1] - points[left]
    point = points[left] + (coordinate - left) * delta
    heading = math.degrees(math.atan2(delta[1], delta[0]))
    return point, heading


class RouteTrackingReward:
    def __init__(self, weights, scales, completion_tolerance, terminal_bonus):
        self.weights = dict(weights)
        self.scales = dict(scales)
        self.tolerance = dict(completion_tolerance)
        self.terminal_bonus = float(terminal_bonus)
        if set(weights) != set(METRICS) or any(
            not math.isfinite(value) or value < 0 for value in weights.values()
        ) or not math.isclose(sum(weights.values()), 1.0):
            raise ValueError("六项奖励权重必须为非负数且总和为 1")
        for values, keys in ((scales, METRICS), (completion_tolerance, ERROR_KEYS)):
            if set(values) != set(keys) or any(
                not math.isfinite(value) or value <= 0 for value in values.values()
            ):
                raise ValueError("奖励尺度和完成阈值必须是齐全的正数")
        if not math.isfinite(self.terminal_bonus) or self.terminal_bonus < 0:
            raise ValueError("结束奖惩必须是非负数")

    def measure(self, aircraft, route, reference, relative_heading, action, previous_action):
        forward, right, _ = ned_to_route_frame(
            aircraft.position_ned_m, route.origin_position_ned_m, route.origin_heading_deg
        )
        altitude_error = abs(aircraft.altitude_m - reference[2])
        speed_error = abs(aircraft.speed_mps - reference[3])
        heading_error = abs(math.remainder(
            aircraft.heading_deg - route.origin_heading_deg - relative_heading, 360.0
        ))
        energy_error = abs(
            9.80665 * (aircraft.altitude_m - reference[2])
            + 0.5 * (aircraft.speed_mps**2 - reference[3]**2)
        )
        action_change = np.abs(action - previous_action).sum()
        values = dict(position=math.hypot(forward - reference[0], right - reference[1]),
                      heading=heading_error, altitude=altitude_error, speed=speed_error,
                      energy=energy_error, action_change=float(action_change))
        reward = 1.0 - sum(self.weights[key] * min(values[key] / self.scales[key], 1.0)
                           for key in METRICS)
        return reward, dict(zip(ERROR_KEYS, (values[key] for key in METRICS[:4])))

    def completed(self, metrics):
        return all(metrics[key] <= limit for key, limit in self.tolerance.items())
