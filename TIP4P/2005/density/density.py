import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


def main():
    # 读取数据文件
    data = np.loadtxt("density.out")

    # 提取数据列：步数 | 温度 | 密度
    if data.shape[1] == 3:
        steps, temperatures, densities = data.T
    else:  # 兼容格式
        temperatures = data[:, 0]
        densities = data[:, 1]

    # 转换单位为°C
    celsius = temperatures - 273.15

    # 创建图表
    plt.figure(figsize=(10, 6))

    # 原始数据点
    plt.plot(celsius, densities, "o", markersize=4, alpha=0.5, label="原始数据点", color="gray")

    # 滑动平均平滑
    window_size = 21
    smoothed_density = np.convolve(densities, np.ones(window_size) / window_size, mode="valid")
    smoothed_celsius = np.convolve(celsius, np.ones(window_size) / window_size, mode="valid")

    # 线性回归拟合密度趋势
    slope, intercept, r_value, p_value, std_err = stats.linregress(temperatures, densities)
    trend_line = slope * temperatures + intercept

    # 绘制平滑曲线和趋势线
    plt.plot(smoothed_celsius, smoothed_density, "b-", linewidth=2, label="平滑密度曲线")
    plt.plot(celsius, trend_line, "k--", alpha=0.7, label="线性趋势")

    # 标记最大密度点
    if len(smoothed_density) > 0:
        max_idx = np.argmax(smoothed_density)
        max_temp_c = smoothed_celsius[max_idx]
        max_density = smoothed_density[max_idx]

        plt.plot(max_temp_c, max_density, "ro", markersize=8)
        plt.text(
            max_temp_c + 1,
            max_density - 1,
            f"density max: {max_temp_c:.1f}°C\n density: {max_density:.2f} kg/m³",
            fontsize=12,
            bbox=dict(facecolor="white", alpha=0.8),
        )

    # 理论4°C参考线
    plt.axvline(4, color="g", linestyle="--", alpha=0.7, label="experiment (4°C)")

    # 标记水的冰点
    plt.axvline(0, color="r", linestyle=":", alpha=0.7, label="ice point(0°C)")

    # 计算统计值
    room_temp = 25
    room_temp_idx = np.argmin(np.abs(celsius - room_temp))
    room_density = densities[room_temp_idx]

    # 添加统计框
    stats_text = (
        f"avg density: {np.mean(densities):.2f} g/cm³\n"
        f"density range: {min(densities):.2f}-{max(densities):.2f} g/cm³\n"
        f"room temperature(25°C)density: {room_density:.2f} g/cm³"
    )

    plt.annotate(
        stats_text,
        xy=(0.02, 0.95),
        xycoords="axes fraction",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    # 设置图表属性
    plt.xlabel("temperature (°C)", fontsize=14)
    plt.ylabel("density (g/cm³)", fontsize=14)
    plt.title("density vs temperature", fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10, loc="lower right")

    # 设置合理的显示范围
    plt.xlim(min(celsius) - 2, max(celsius) + 2)
    plt.ylim(min(densities) - 5, max(densities) + 5)

    # 保存图像
    plt.tight_layout()
    # plt.savefig("density_vs_temperature.png", dpi=300)
    plt.show()
    print(f"图像已保存为 density_vs_temperature.png")


if __name__ == "__main__":
    main()
