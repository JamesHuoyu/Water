import numpy as np
import matplotlib.pyplot as plt

# 生成概念性的数据来展示各个阶段
t = np.logspace(-3, 3, 500)  # 时间从0.001 ps到1000 ps

# 1. 弹道区 - 二次方增长
msd_ballistic = 100 * t**2
# 2. 平台区 - 模拟cage效应，增长极其缓慢
# 使用一个平滑的阶梯函数来模拟
from scipy.special import expit

plateau_time = 0.1
plateau_length = 10
msd_plateau = 1.5 + 0.5 * expit((np.log10(t) - np.log10(plateau_time)) * 10)
# 3. 亚扩散区 - 幂律增长
alpha = 0.6  # 亚扩散指数
msd_subdiffusive = 5 * t**alpha
# 4. 扩散区 - 线性增长
msd_diffusive = 0.01 * t

# 将四个阶段平滑地连接起来形成一条连续的MSD曲线
# 使用权重函数进行混合
w_ball = expit(-(np.log10(t) + 1.5) * 5)
w_plat = expit(-(np.log10(t) - np.log10(plateau_time)) * 5) * expit(
    (np.log10(t) - np.log10(plateau_length)) * 5
)
w_sub = expit(-(np.log10(t) - np.log10(plateau_length)) * 5) * expit((np.log10(t) - 1.5) * 5)
w_diff = expit((np.log10(t) - 1.5) * 5)

msd_total = (
    w_ball * msd_ballistic
    + w_plat * msd_plateau
    + w_sub * msd_subdiffusive
    + w_diff * msd_diffusive
)

# 绘制图像
plt.figure(figsize=(10, 6))
plt.loglog(t, msd_total, "b-", linewidth=3, label="概念性MSD曲线")

# 标注各个区域
plt.axvline(x=plateau_time, color="k", linestyle="--", alpha=0.5)
plt.axvline(x=plateau_length, color="k", linestyle="--", alpha=0.5)
plt.axvline(x=10**1.5, color="k", linestyle="--", alpha=0.5)  # ~32 ps

plt.text(
    1e-2,
    1e-1,
    "I. 弹道区\n(斜率 ≈ 2)",
    ha="center",
    va="center",
    fontsize=12,
    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
)
plt.text(
    1e0,
    2.5,
    "II. 平台区\n(~1 Å²)",
    ha="center",
    va="center",
    fontsize=12,
    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
)
plt.text(
    1e1,
    1e1,
    "III. 亚扩散区\n(斜率 < 1)",
    ha="center",
    va="center",
    fontsize=12,
    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
)
plt.text(
    1e3,
    1e2,
    "IV. 正常扩散区\n(斜率 = 1)",
    ha="center",
    va="center",
    fontsize=12,
    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
)

plt.xlabel("时间 (ps)")
plt.ylabel("MSD (Å²)")
plt.title("TIP4P/2005水模型在230K的MSD概念图 (双对数坐标)")
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()
