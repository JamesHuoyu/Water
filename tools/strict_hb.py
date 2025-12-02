import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


results = pd.read_csv("test_hbond_counts.csv").to_numpy()
classification1 = np.load("/home/debian/water/TIP4P/2005/2020/rst/equili/classification1.npy")
classification2 = np.load("/home/debian/water/TIP4P/2005/2020/rst/equili/classification2.npy")


def compute_number_of_ideal_hbonds(classification):
    """
    计算属于目标类别的分子数，按frame进行统计
    """
    target_numbers = np.sum(classification, axis=0)
    print(target_numbers.shape)
    return target_numbers


def plot_ideal_hbonds_vs_time(classification1, classification2):
    """
    绘制理想氢键数随时间变化的图表
    """
    time_length = np.shape(classification1)[1]  # 假设第一列是时间
    time = np.arange(time_length) * 0.2  # 转换为皮秒，假设时间步长为0.2 ps
    ideal_hbonds1 = compute_number_of_ideal_hbonds(classification1)
    ideal_hbonds2 = compute_number_of_ideal_hbonds(classification2)

    plt.figure(figsize=(10, 6))
    plt.plot(time, ideal_hbonds1, label="Ideal HBonds Type 1", color="blue")
    plt.plot(time, ideal_hbonds2, label="Ideal HBonds Type 2", color="orange")
    plt.xlabel("Time (ps)")
    plt.ylabel("Number of Ideal Hydrogen Bonds")
    plt.title("Number of Ideal Hydrogen Bonds vs Time")
    plt.legend()
    plt.grid()
    plt.show()


# 调用函数绘图
plot_ideal_hbonds_vs_time(classification1, classification2)
# temp = compute_number_of_ideal_hbonds(classification1)
