import os
import yaml
import pymap3d
import numpy as np


def parse_config(filename, policy_type, fix_position):
    """Parse JSBSim config file.

    Args:
        config (str): config file name

    Returns:
        (EnvConfig): a custom class which parsing dict into object.
    """
    filepath = os.path.join(get_root_dir(), 'configs', f'{filename}.yaml')
    assert os.path.exists(filepath), \
        f'config path {filepath} does not exist. Please pass in a string that represents the file path to the config yaml.'
    with open(filepath, 'r', encoding='utf-8') as f:
        config_data = yaml.load(f, Loader=yaml.FullLoader)
    config_data['policy_type'] = policy_type
    config_data['fix_position'] = fix_position
    return type('EnvConfig', (object,), config_data)


def get_root_dir():
    return os.path.join(os.path.split(os.path.realpath(__file__))[0], '..')


def LLA2NEU(lon, lat, alt, lon0=120.0, lat0=60.0, alt0=0):
    """Convert from Geodetic Coordinate System to NEU Coordinate System.

    Args:
        lon, lat, alt (float): target geodetic lontitude(°), latitude(°), altitude(m)
        lon, lat, alt (float): observer geodetic lontitude(°), latitude(°), altitude(m); Default=`(120°E, 60°N, 0m)`

    Returns:
        (np.array): (North, East, Up), unit: m
    """
    n, e, d = pymap3d.geodetic2ned(lat, lon, alt, lat0, lon0, alt0)
    return np.array([n, e, -d])


def NEU2LLA(n, e, u, lon0=120.0, lat0=60.0, alt0=0):
    """Convert from NEU Coordinate System to Geodetic Coordinate System.

    Args:
        n, e, u (float): target relative position w.r.t. North, East, Down
        lon, lat, alt (float): observer geodetic lontitude(°), latitude(°), altitude(m); Default=`(120°E, 60°N, 0m)`

    Returns:
        (np.array): (lon, lat, alt), unit: °, °, m
    """
    lat, lon, h = pymap3d.ned2geodetic(n, e, -u, lat0, lon0, alt0)
    return np.array([lon, lat, h])


def get_AO_TA_R(ego_feature, enm_feature, return_side=False):
    """Get AO & TA angles and relative distance between two agent.

    Args:
        ego_feature & enemy_feature (tuple): (north, east, down, vn, ve, vd)

    Returns:
        (tuple): ego_AO, ego_TA, R
    """
    ego_x, ego_y, ego_z, ego_vx, ego_vy, ego_vz = ego_feature
    ego_v = np.linalg.norm([ego_vx, ego_vy, ego_vz])
    enm_x, enm_y, enm_z, enm_vx, enm_vy, enm_vz = enm_feature
    enm_v = np.linalg.norm([enm_vx, enm_vy, enm_vz])
    delta_x, delta_y, delta_z = enm_x - ego_x, enm_y - ego_y, enm_z - ego_z
    R = np.linalg.norm([delta_x, delta_y, delta_z])

    proj_dist = delta_x * ego_vx + delta_y * ego_vy + delta_z * ego_vz
    ego_AO = np.arccos(np.clip(proj_dist / (R * ego_v + 1e-8), -1, 1))
    proj_dist = delta_x * enm_vx + delta_y * enm_vy + delta_z * enm_vz
    ego_TA = np.arccos(np.clip(proj_dist / (R * enm_v + 1e-8), -1, 1))

    if not return_side:
        return ego_AO, ego_TA, R
    else:
        side_flag = np.sign(np.cross([ego_vx, ego_vy], [delta_x, delta_y]))
        return ego_AO, ego_TA, R, side_flag


def get2d_AO_TA_R(ego_feature, enm_feature, return_side=False):
    ego_x, ego_y, ego_z, ego_vx, ego_vy, ego_vz = ego_feature
    ego_v = np.linalg.norm([ego_vx, ego_vy])
    enm_x, enm_y, enm_z, enm_vx, enm_vy, enm_vz = enm_feature
    enm_v = np.linalg.norm([enm_vx, enm_vy])
    delta_x, delta_y, delta_z = enm_x - ego_x, enm_y - ego_y, enm_z - ego_z
    R = np.linalg.norm([delta_x, delta_y])

    proj_dist = delta_x * ego_vx + delta_y * ego_vy
    ego_AO = np.arccos(np.clip(proj_dist / (R * ego_v + 1e-8), -1, 1))
    proj_dist = delta_x * enm_vx + delta_y * enm_vy
    ego_TA = np.arccos(np.clip(proj_dist / (R * enm_v + 1e-8), -1, 1))

    if not return_side:
        return ego_AO, ego_TA, R
    else:
        side_flag = np.sign(np.cross([ego_vx, ego_vy], [delta_x, delta_y]))
        return ego_AO, ego_TA, R, side_flag


def in_range_deg(angle):
    """ Given an angle in degrees, normalises in (-180, 180] """
    angle = angle % 360
    if angle > 180:
        angle -= 360
    return angle


def in_range_rad(angle):
    """ Given an angle in rads, normalises in (-pi, pi] """
    angle = angle % (2 * np.pi)
    if angle > np.pi:
        angle -= 2 * np.pi
    return angle


def angle_diff(angle1, angle2):
    """ 得到angle1到angle2的最小角度，范围在(-pi, pi] """
    return np.arctan2(np.sin(angle1-angle2), np.cos(angle1-angle2))


def get_center_of_multi_air(ego_position: np.array, enm_positions: np.array):
    """ 在多机环境下，获取敌机相对我机的加权质心和我机到质心的向量 """
    # 计算向量
    vectors = enm_positions - ego_position

    # 计算每个向量的欧氏距离
    distances = np.linalg.norm(vectors, axis=1)

    # 避免除零（如果有点和p0重合）
    epsilon = 1e-8
    distances = np.maximum(distances, epsilon)

    # 距离越远权重越小，例如使用 1/distance
    weights = 1 / distances ** 2

    # 归一化权重
    weights /= weights.sum()

    # 计算加权质心向量
    centroid_vector = np.sum(vectors * weights[:, np.newaxis], axis=0)

    # 如果需要质心的绝对坐标
    centroid_position = ego_position + centroid_vector

    return centroid_position, centroid_vector

def get_near_offset_of_multi_air(ego_position: np.array, enm_positions: np.array, lamda: float = 0.3):
    """ 在多机环境下，获取以最近敌机为锚点，远离多机质心一侧的函数 """
    # 获取指向最近敌机的向量
    distances = np.linalg.norm(enm_positions - ego_position, axis=1)
    min_idx = np.argmin(distances)
    nearest_vector = enm_positions[min_idx] - ego_position

    # # 获取指向敌方质心的向量
    # _, center_vector = get_center_of_multi_air(ego_position, enm_positions)
    #
    # # 计算以最近敌机为锚点的偏移量
    # nearest2center = center_vector - nearest_vector
    # nearest_direction = -nearest_vector / np.linalg.norm(nearest_vector)
    # offset = nearest2center - (np.dot(nearest2center, nearest_direction) * nearest_direction)
    #
    # # 获得指向偏移后的坐标点的向量
    # target_vector = nearest_vector - lamda * offset
    # target_position = ego_position + target_vector
    #
    # return target_position, target_vector
    return enm_positions[min_idx], nearest_vector