import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.optimize import curve_fit

nhb = pd.read_csv("quenching/nhb_min_distances.csv").rename(columns={"min_distance": "nhb"})
hb = pd.read_csv("quenching/max_distance_per_idx.csv").rename(
    columns={"max_distance": "hb", "idx": "O_idx"}
)
df = pd.merge(nhb, hb, on=["frame", "O_idx"], how="outer")
# 如果原数据是 Å -> 转为 nm
df["nhb_nm"] = df["nhb"] / 10.0
df["hb_nm"] = df["hb"] / 10.0
df = df.dropna(subset=["nhb_nm", "hb_nm"])
df["z"] = df["nhb_nm"] - df["hb_nm"]

# 1) KDE with several bandwidths
zs = df["z"].values
bw_list = [0.001, 0.0025, 0.005, 0.01]  # nm
x = np.linspace(zs.min() - 0.02, zs.max() + 0.02, 1000)

plt.figure(figsize=(8, 5))
for bw in bw_list:
    kde = gaussian_kde(zs, bw_method=bw / np.std(zs))
    plt.plot(x, kde(x), label=f"bandwidth={bw} nm")
plt.axvline(0, color="k", linestyle="--")
plt.legend()
plt.xlabel("z (nm)")
plt.title("KDE with different bandwidths")
plt.savefig("quenching/kde_bandwidth_comparison.png", dpi=300)
plt.show()

# 2) Histogram sensitivity
plt.figure(figsize=(8, 5))
for bins in [30, 60, 120]:
    plt.hist(zs, bins=bins, density=True, alpha=0.4, label=f"bins={bins}")
plt.legend()
plt.axvline(0, color="k", ls="--")
plt.xlabel("z (nm)")
plt.title("Histogram with different bin sizes")
plt.savefig("quenching/histogram_bin_sensitivity.png", dpi=300)
plt.show()

# 3) GMM 两个高斯拟合（验证是否可用两高斯描述）


def two_gaussians(x, w1, mu1, sigma1, mu2, sigma2):
    w2 = 1 - w1
    return w1 * np.exp(-0.5 * ((x - mu1) / sigma1) ** 2) / (
        sigma1 * np.sqrt(2 * np.pi)
    ) + w2 * np.exp(-0.5 * ((x - mu2) / sigma2) ** 2) / (sigma2 * np.sqrt(2 * np.pi))


# initial guess
p0 = [0.5, -0.01, 0.005, 0.01, 0.005]
hist, bin_edges = np.histogram(zs, bins=100, density=True)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
params, cov = curve_fit(two_gaussians, bin_centers, hist, p0=p0)
w1, mu1, sigma1, mu2, sigma2 = params
print(
    f"Fitted parameters:\nw1={w1:.3f}, mu1={mu1:.3f}, sigma1={sigma1:.3f}\nmu2={mu2:.3f}, sigma2={sigma2:.3f}"
)
# plot fit
plt.figure(figsize=(8, 5))
plt.hist(zs, bins=100, density=True, alpha=0.3, label="Data histogram")
x_fit = np.linspace(zs.min(), zs.max(), 1000)
plt.plot(x_fit, two_gaussians(x_fit, *params), "-k", lw=2, label="Two-Gaussian Fit")
# individual gaussians
from scipy.stats import norm

plt.plot(x_fit, w1 * norm.pdf(x_fit, mu1, sigma1), "--", label=f"Comp 1 (mu={mu1:.3f})")
plt.plot(x_fit, (1 - w1) * norm.pdf(x_fit, mu2, sigma2), "--", label=f"Comp 2 (mu={mu2:.3f})")
plt.axvline(0, color="k", ls="--")
plt.legend()
plt.xlabel("z (nm)")
plt.title("Two-Gaussian Fit to Histogram")
plt.savefig("quenching/two_gaussian_fit.png", dpi=300)
plt.show()


# 4) 分时间段检查（例如每 20 ns 一段，假设 frame -> time map available）
# 如果没有时间列，可用 frame index and dt to map to ns.
# 假设 df 已有 'time_ns' 列:
if "time_ns" in df.columns:
    for t0, t1 in [(0, 20), (20, 40), (40, 60), (60, 80), (80, 100), (100, 120)]:
        sel = df[(df["time_ns"] >= t0) & (df["time_ns"] < t1)]["z"]
        if len(sel) > 1000:
            plt.hist(sel, bins=80, density=True, alpha=0.5, label=f"{t0}-{t1}ns")
plt.legend()
plt.show()
