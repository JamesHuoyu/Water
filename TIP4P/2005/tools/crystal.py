import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from sklearn.mixture import GaussianMixture

nhb = pd.read_csv("rst/nhb_min_distances.csv").rename(columns={"min_distance": "nhb"})
hb = pd.read_csv("rst/max_distance_per_idx.csv").rename(
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
plt.show()

# 2) Histogram sensitivity
plt.figure(figsize=(8, 5))
for bins in [30, 60, 120]:
    plt.hist(zs, bins=bins, density=True, alpha=0.4, label=f"bins={bins}")
plt.legend()
plt.axvline(0, color="k", ls="--")
plt.xlabel("z (nm)")
plt.show()

# 3) GMM 两个高斯拟合（验证是否可用两高斯描述）
z_reshape = zs.reshape(-1, 1)
gmm = GaussianMixture(n_components=2, covariance_type="full", random_state=0).fit(z_reshape)
weights = gmm.weights_
means = gmm.means_.flatten()
covs = gmm.covariances_.flatten()
print("GMM components:", weights, means, np.sqrt(covs))
# plot GMM components + data KDE
x = np.linspace(zs.min(), zs.max(), 1000)
pdf_gmm = np.exp(gmm.score_samples(x.reshape(-1, 1)))
plt.figure(figsize=(8, 5))
plt.hist(zs, bins=100, density=True, alpha=0.3)
plt.plot(x, pdf_gmm, "-k", lw=2, label="GMM mixture")
# individual gaussians
from scipy.stats import norm

for w, m, cv in zip(weights, means, covs):
    plt.plot(x, w * norm.pdf(x, m, np.sqrt(cv)), "--", label=f"comp m={m:.3f}")
plt.axvline(0, color="k", ls="--")
plt.legend()
plt.xlabel("z (nm)")
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
