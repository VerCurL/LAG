"""决策时刻机体对齐局部航迹坐标的转换函数。"""

import math
from typing import Tuple


def _rotation(heading_deg: float) -> Tuple[float, float]:
    """计算局部航迹坐标系所需的航向旋转量。"""
    angle = math.radians(heading_deg)
    return math.cos(angle), math.sin(angle)


def ned_to_route_frame(
    position_ned_m: Tuple[float, float, float],
    origin_ned_m: Tuple[float, float, float],
    origin_heading_deg: float,
) -> Tuple[float, float, float]:
    """把 NED 位置转换为相对原点的前向、右向和向下坐标。"""
    dn = position_ned_m[0] - origin_ned_m[0]
    de = position_ned_m[1] - origin_ned_m[1]
    dd = position_ned_m[2] - origin_ned_m[2]
    c, s = _rotation(origin_heading_deg)
    return (c * dn + s * de, -s * dn + c * de, dd)


def route_frame_to_ned(
    point: Tuple[float, float, float],
    origin_ned_m: Tuple[float, float, float],
    origin_heading_deg: float,
) -> Tuple[float, float, float]:
    """把前向、右向和向下坐标转换回 NED 位置。"""
    forward, right, down = point
    c, s = _rotation(origin_heading_deg)
    return (
        origin_ned_m[0] + c * forward - s * right,
        origin_ned_m[1] + s * forward + c * right,
        origin_ned_m[2] + down,
    )


def body_aligned_route_point(
    position_ned_m: Tuple[float, float, float],
    altitude_m: float,
    speed_mps: float,
    origin_ned_m: Tuple[float, float, float],
    origin_heading_deg: float,
) -> Tuple[float, float, float, float]:
    """根据位置、高度和速度生成机体对齐坐标中的航迹点。"""
    forward, right, _down = ned_to_route_frame(
        position_ned_m, origin_ned_m, origin_heading_deg
    )
    return (forward, right, altitude_m, speed_mps)
