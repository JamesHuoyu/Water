import re
import numpy as np
import matplotlib.pyplot as plt


def extract_pxy_from_log(log_file, output_txt="pxy_data.txt"):
    """
    从LAMMPS log文件中提取timestep和Pxy数据

    参数:
    log_file: LAMMPS log文件路径
    output_txt: 输出txt文件路径
    """

    # 用于存储提取的数据
    timesteps = []
    pxy_values = []

    # 编译正则表达式，匹配数据行
    # 匹配类似这样的行: "     100   227.64461     -52548.587     -58106.716     -358.17217      0.038076637    0.005          0.00076153273"
    # 我们需要第1列(Step)和第5列(Pxy)
    data_pattern = re.compile(
        r"^\s*(\d+)\s+[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?\s+[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?\s+[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s+[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?\s+[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?\s+[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?\s*$"
    )

    # 编译一个更简单的正则表达式版本，用于更灵活匹配
    # 匹配: 数字 空格+ 数字 空格+ 数字 空格+ 数字 空格+ 数字（第5列，Pxy）
    simple_pattern = re.compile(r"^\s*(\d+)\s+\S+\s+\S+\s+\S+\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")

    with open(log_file, "r") as f:
        for line in f:
            # 使用正则表达式匹配
            match = data_pattern.match(line)
            if not match:
                # 尝试简单模式匹配
                match = simple_pattern.match(line)

            if match:
                try:
                    timestep = int(match.group(1))
                    pxy = float(match.group(2))

                    timesteps.append(timestep)
                    pxy_values.append(pxy)
                except ValueError:
                    continue

    # 写入txt文件
    with open(output_txt, "w") as f:
        f.write("# Step Pxy\n")
        for ts, pxy in zip(timesteps, pxy_values):
            f.write(f"{ts} {pxy}\n")

    print(f"提取了 {len(timesteps)} 个数据点")
    print(f"数据已保存到 {output_txt}")

    return np.array(timesteps), np.array(pxy_values)


def plot_viscosity(timesteps, viscosities, shear_rate, output_png="viscosity_plot.png"):
    """绘制viscosity随时间的变化图"""

    # 创建图形
    plt.figure(figsize=(12, 6))

    # 计算strain (假设从0开始，每1000步增加0.005)
    # 从你的数据看，v_strain每1000步增加0.005
    strain = shear_rate * timesteps

    # 绘制viscosity随timestep的变化
    plt.subplot(1, 2, 1)
    plt.plot(timesteps, viscosities, "b-", linewidth=1, alpha=0.7)
    plt.plot(timesteps, viscosities, "r.", markersize=3)
    plt.xlabel("Timestep")
    plt.ylabel("v_viscosity")
    plt.title("Viscosity vs Timestep")
    plt.grid(True, alpha=0.3)

    # 绘制viscosity随strain的变化
    plt.subplot(1, 2, 2)
    plt.plot(strain, viscosities, "b-", linewidth=1, alpha=0.7)
    plt.plot(strain, viscosities, "r.", markersize=3)
    plt.xlabel("Strain")
    plt.ylabel("v_viscosity")
    plt.title("Viscosity vs Strain")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_png, dpi=300)
    print(f"图形已保存到 {output_png}")
    plt.show()

    # 打印统计信息
    print("\n统计信息:")
    print(f"数据点数: {len(viscosities)}")
    print(f"Timestep范围: {timesteps[0]} - {timesteps[-1]}")
    print(f"Viscosity范围: {viscosities.min():.6f} - {viscosities.max():.6f}")
    print(f"Viscosity平均值: {viscosities.mean():.6f} ± {viscosities.std():.6f}")
    print(f"Viscosity中位数: {np.median(viscosities):.6f}")


# 使用更简单但更可靠的方法提取数据
def extract_viscosity_simple(log_file, output_txt="viscosity_data.txt"):
    """使用简单方法提取数据"""

    timesteps = []
    viscosities = []

    with open(log_file, "r") as f:
        for line in f:
            # 去掉行首尾空白
            line = line.strip()

            # 跳过空行
            if not line:
                continue

            # 检查是否以数字开头（数据行）
            if line[0].isdigit():
                # 分割行
                parts = line.split()

                # 检查是否有足够的部分（至少8列）
                if len(parts) >= 8:
                    try:
                        timestep = int(parts[0])
                        viscosity = float(parts[-1])  # v_viscosity是最后一列

                        timesteps.append(timestep)
                        viscosities.append(viscosity)
                    except ValueError:
                        # 如果转换失败，跳过这行
                        continue

    # 确保数据按timestep排序
    sorted_data = sorted(zip(timesteps, viscosities))
    timesteps = [d[0] for d in sorted_data]
    viscosities = [d[1] for d in sorted_data]

    # 写入txt文件
    with open(output_txt, "w") as f:
        f.write("# Timestep  v_viscosity\n")
        for ts, visc in zip(timesteps, viscosities):
            f.write(f"{ts}  {visc:.10f}\n")

    print(f"提取了 {len(timesteps)} 个数据点")
    print(f"数据已保存到 {output_txt}")

    return np.array(timesteps), np.array(viscosities)


# 如果需要处理多个log文件
def extract_from_multiple_logs(log_files, output_txt="combined_viscosity.txt"):
    """从多个log文件提取并合并数据"""

    all_timesteps = []
    all_viscosities = []

    for i, log_file in enumerate(log_files):
        print(f"处理文件 {i+1}/{len(log_files)}: {log_file}")
        timesteps, viscosities = extract_viscosity_simple(log_file, f"temp_{i}.txt")

        # 如果需要，可以在这里添加偏移
        all_timesteps.extend(timesteps)
        all_viscosities.extend(viscosities)

    # 排序
    sorted_data = sorted(zip(all_timesteps, all_viscosities))
    timesteps = [d[0] for d in sorted_data]
    viscosities = [d[1] for d in sorted_data]

    # 保存合并的数据
    with open(output_txt, "w") as f:
        f.write("# Timestep  v_viscosity\n")
        for ts, visc in zip(timesteps, viscosities):
            f.write(f"{ts}  {visc:.10f}\n")

    print(f"\n合并了 {len(log_files)} 个文件，总共 {len(timesteps)} 个数据点")
    print(f"合并数据已保存到 {output_txt}")

    return np.array(timesteps), np.array(viscosities)


# 主程序
if __name__ == "__main__":
    # 指定你的log文件路径
    log_file_path = "/home/debian/water/TIP4P/Ice/test/log_5e-6_225.lammps"  # 替换为你的log文件路径

    try:
        # 方法1: 使用简单方法提取
        print("使用简单方法提取数据...")
        timesteps, pxy_values = extract_pxy_from_log(log_file_path, "pxy_data.txt")

        # # 绘制图形
        # plot_viscosity(timesteps, pxy_values, 5e-6, "viscosity_plot5e-6.png")

        # # 保存为numpy格式以便后续处理
        # np.savez("viscosity_data.npz", timesteps=timesteps, viscosities=viscosities)
        # print("数据已保存为 viscosity_data.npz")

    except FileNotFoundError:
        print(f"错误: 找不到文件 {log_file_path}")
        print("请确保log文件路径正确")

        # 示例: 如果文件不存在，创建一个示例输出
        print("\n创建示例输出文件...")
        with open("viscosity_data.txt", "w") as f:
            f.write("# Timestep  v_viscosity\n")
            # 示例数据
            for i in range(0, 300001, 1000):
                viscosity = 0.01 + 0.005 * np.sin(i / 10000) + 0.001 * np.random.randn()
                f.write(f"{i}  {viscosity:.10f}\n")
        print("已创建示例文件 viscosity_data.txt")
