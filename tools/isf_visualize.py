import matplotlib.pyplot as plt
import pandas as pd
import json
import numpy as np
from scipy.optimize import curve_fit


def calculate_tau_alpha(time_ps, isf, threshold=1 / np.e):
    """计算tau_alpha，即ISF衰减到1/e的时间"""
    for t, f in zip(time_ps, isf):
        if f <= threshold:
            return t
    return np.nan


def KWW_fit_func(t, A, tau, beta):
    return A * np.exp(-((t / tau) ** beta))


def plot_isf(filepath: str, key: str, start_time: bool = True):
    df = pd.read_hdf(filepath, key=key)
    if start_time:
        x = df["time_ps"][1:]
        y = df["ISF"][1:]
    else:
        x = df["time_ps"]
        y = df["ISF"]
    plt.plot(x, y, "-", markersize=4, label=key)
    tau_alpha_ps = calculate_tau_alpha(x, y)
    print(f"{key}: tau_alpha = {tau_alpha_ps:.5f} ps")
    # # KWW拟合
    # try:
    #     popt, pcov = curve_fit(
    #         KWW_fit_func,
    #         x,
    #         y,
    #         p0=(1.0, tau_alpha_ps, 0.8),
    #         bounds=([0, 0, 0], [1.0, np.inf, 2.0]),
    #     )
    #     A_fit, tau_fit, beta_fit = popt
    #     print(f"  KWW fit: A={A_fit:.4f}, tau={tau_fit:.4f} ps, beta={beta_fit:.4f}")
    #     x_fit = np.linspace(min(x), max(x), 200)
    #     y_fit = KWW_fit_func(x_fit, *popt)
    #     plt.plot(x_fit, y_fit, "--", label=f"KWW fit ({key})")
    # except Exception as e:
    #     print(f"  KWW fit failed for {key}: {e}")


# 示例使用
plt.figure(figsize=(7, 5))
# plot_isf("/home/debian/water/TIP4P/2005/2020/rst/4096/all_isf_results.h5", "2.5e-4")
plot_isf("/home/debian/water/TIP4P/2005/2020/rst/4096/all_isf_results.h5", "2.5e-5")
plot_isf("test_isf_results.h5", "2.5e-4-everystep")
# plot_isf("test_isf_results.h5", "2.5e-5-everystep")
plot_isf("/home/debian/water/TIP4P/2005/Tanaka_2018/rst/equili_isf_results.h5", "H2O")
plt.xlabel("Time (ps)")
plt.xscale("log")
plt.ylabel(r"$F_s(k,t)$")
plt.title("Self-intermediate Scattering Function")
plt.ylim(0, 1)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
