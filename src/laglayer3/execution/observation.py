
"""将固定航迹坐标系中的点列和飞机状态编码为执行模型输入。"""

import math

import numpy as np

from laglayer3.planning.frames import ned_to_route_frame

STATE_DIM = 16
ROUTE_FEATURES = 5
OBSERVATION_SCHEMA = "laglayer3.execution_observation.v1"


def encode_execution_observation(aircraft, route, elapsed, previous_action, window_size):
    origin_altitude = -route.origin_position_ned_m[2]
    position = ned_to_route_frame(aircraft.position_ned_m, route.origin_position_ned_m,
                                  route.origin_heading_deg)
    velocity = ned_to_route_frame(aircraft.velocity_ned_mps, (0.0, 0.0, 0.0),
                                  route.origin_heading_deg)
    relative_heading = math.radians(aircraft.heading_deg - route.origin_heading_deg)
    point_index, phase = divmod(round(elapsed / route.interval_seconds, 10), 1.0)
    first = min(int(point_index), len(route.points))
    window = np.zeros((window_size, ROUTE_FEATURES), dtype=np.float32)
    points = np.asarray(route.points[first:first + window_size], dtype=np.float32)
    if len(points):
        window[:len(points), :4] = (
            points - np.array([0, 0, origin_altitude, 0], dtype=np.float32)
        ) / np.array([5000, 5000, 1000, 300], dtype=np.float32)
        window[:len(points), 4] = 1
    state = np.asarray([
        position[0] / 5000, position[1] / 5000, position[2] / 1000,
        *(component / 300 for component in velocity),
        math.sin(relative_heading), math.cos(relative_heading),
        aircraft.altitude_m / 10000, aircraft.speed_mps / 300,
        aircraft.metadata.get("roll_rad", 0.0), aircraft.metadata.get("pitch_rad", 0.0),
        phase,
        previous_action[0] - 1, (previous_action[1] - 2) / 2, previous_action[2] - 1,
    ], dtype=np.float32)
    return window, state
