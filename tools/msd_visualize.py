import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


def cal_diffusion_coefficient(time_ps, msd):
    """计算扩散系数D，假设MSD = 2nDt，其中n为维度（这里n=1）"""
    # 只使用线性部分进行拟合，假设后50%的数据为线性部分
    n_points = len(time_ps)
    start_index = int(n_points * 0.5)
    x_fit = time_ps[start_index:]
    y_fit = msd[start_index:]
    popt, _ = curve_fit(lambda t, D: 2 * 1 * D * t, x_fit, y_fit)
    D = popt[0]
    # 计算拟合优度R²
    residuals = y_fit - (2 * 1 * D * x_fit)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y_fit - np.mean(y_fit)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    print(f"Fitted Diffusion Coefficient D = {D:.5e} Å²/ps, R² = {r_squared:.5f}")
    return D, r_squared, start_index


def plot_msd(filepath: str, key: str):
    df = pd.read_hdf(filepath, key=key)
    x = df["time_ps"][1:]
    y = df["MSD_A2"][1:]
    # print(f"MSD Analysis for {key}:")
    # D, r2, fit_start = cal_diffusion_coefficient(x, y)
    plt.plot(x, y, "-", markersize=4, label=key)
    # plt.axvline(x=x[fit_start], color="gray", linestyle="--", label="Fit Start")


# 示例使用
plt.figure(figsize=(7, 5))
plot_msd("equili_msd_results.h5", "246")
plot_msd("equili_msd_results.h5", "246-x")
plot_msd("/home/debian/water/TIP4P/2005/2020/rst/4096/all_msd_results.h5", "2.5e-4")
plot_msd("/home/debian/water/TIP4P/2005/2020/rst/4096/all_msd_results.h5", "2.5e-4-y")
# plot_msd("/home/debian/water/TIP4P/2005/2020/rst/4096/all_msd_results.h5", "2.5e-4-x")
plot_msd("/home/debian/water/TIP4P/2005/2020/rst/4096/all_msd_results.h5", "2.5e-5")
plot_msd("/home/debian/water/TIP4P/2005/2020/rst/4096/all_msd_results.h5", "2.5e-5-y")
# plot_msd("/home/debian/water/TIP4P/2005/2020/rst/4096/all_msd_results.h5", "2.5e-5-x")
# plot_msd("test_msd_results.h5", "2.5e-5-x")
# plot_msd("test_msd_results.h5", "2.5e-4-everystep")
# plot_msd("test_msd_results.h5", "2.5e-5-everystep")
# plot_msd("/home/debian/water/TIP4P/2005/2020/rst/4096/all_msd_results.h5", "2.5e-6")
# plot_msd("/home/debian/water/TIP4P/2005/2020/rst/4096/all_msd_results.h5", "5e-6")
# plot_msd("/home/debian/water/TIP4P/2005/2020/rst/4096/all_msd_results.h5", "5e-7")
plt.xlabel("Time (ps)")
plt.xscale("log")
plt.ylabel("MSD (Å²)")
plt.yscale("log")
plt.title("Mean Squared Displacement")
plt.legend()
plt.grid(True)
plt.show()
