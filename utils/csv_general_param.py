import os
import pandas as pd
import matplotlib.pyplot as plt

# 指定存放 CSV 文件的目录
data_folder = "./results/ppo.vs.ppoMoEB [lc]"

# 自动创建 data_folder（如果不存在）
os.makedirs(data_folder, exist_ok=True)

# 自动创建 figure 子目录
figure_folder = os.path.join(data_folder, "figure")
os.makedirs(figure_folder, exist_ok=True)

# 你关心的列
columns_to_plot = [
    "policy_loss",
    "policy_entropy_loss",
    "value_loss",
    "average_episode_rewards",
    "fps"
]

episode_col = "episode"

# 找到所有 CSV 文件
csv_files = [f for f in os.listdir(data_folder) if f.endswith(".csv")]
if not csv_files:
    print("未发现 CSV 文件，请检查目录。")
    exit()

print("发现的 CSV 文件：", csv_files)

# 每列画一个图
for col in columns_to_plot:
    plt.figure(figsize=(10, 6))
    plt.title(f"Comparison of {col}")
    plt.xlabel("Episode")
    plt.ylabel(col)

    has_data = False

    for csv_file in csv_files:
        file_path = os.path.join(data_folder, csv_file)

        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            print(f"读取失败: {csv_file}, 错误: {e}")
            continue

        # 检查必要列是否存在
        if episode_col not in df.columns or col not in df.columns:
            print(f"文件 {csv_file} 缺少必要列 '{episode_col}' 或 '{col}'，已跳过。")
            continue

        # 绘制曲线，使用文件名当 label
        plt.plot(df[episode_col], df[col], label=os.path.splitext(csv_file)[0])
        has_data = True

    if has_data:
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        save_path = os.path.join(figure_folder, f"{col}_comparison.png")
        plt.savefig(save_path, dpi=300)
        plt.show()
        print(f"已保存图像: {save_path}")
    else:
        plt.close()
        print(f"没有任何文件包含列 {col}，无法绘图。")

print("绘图完成！所有 PNG 图像均保存在 figure 目录中。")
