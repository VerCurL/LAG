import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def near_offset_positions_plot(ego_position, enm_positions, near_offset_position):
    # 绘图
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 敌机位置
    ax.scatter(enm_positions[:, 0], enm_positions[:, 1], enm_positions[:, 2], c='red', s=100, label='Enemies')

    # 最近敌机偏移位置
    ax.scatter(near_offset_position[0], near_offset_position[1], near_offset_position[2],
               c='blue', s=100, label='Near offset')

    # 我机
    ax.scatter(ego_position[0], ego_position[1], ego_position[2],
               c='green', s=150, label='Ego')

    # 标注点
    for i, pos in enumerate(enm_positions):
        ax.text(pos[0], pos[1], pos[2], f'E{i + 1}', color='red')
    ax.text(near_offset_position[0], near_offset_position[1], near_offset_position[2], 'Near', color='blue')
    ax.text(*ego_position, 'Ego', color='green')

    # Ego -> Near offset
    ax.plot([near_offset_position[0], ego_position[0]],
            [near_offset_position[1], ego_position[1]],
            [near_offset_position[2], ego_position[2]],
            linestyle='--', color='green', alpha=0.7)

    # Ego -> 每个敌机
    for pos in enm_positions:
        ax.plot([pos[0], ego_position[0]],
                [pos[1], ego_position[1]],
                [pos[2], ego_position[2]],
                linestyle=':', color='gray', alpha=0.5)

    # 坐标轴标签
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # 标题和图例
    ax.set_title('Ego / Enemy / Near-offset geometric relation')
    ax.legend()
    ax.view_init(elev=2, azim=-60)  # 可以调整视角

    plt.tight_layout()
    plt.show()