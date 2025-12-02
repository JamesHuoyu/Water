import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import norm


class Classifier:
    def __init__(self, zeta: pd.DataFrame):
        self.zeta = zeta
        # 估计 P(0) - ζ=0 处的概率密度
        self.x, self.kde, self.P0 = self.estimate_P0()
        self.m_rho, self.sigma_rho, self.m_S, self.sigma_S = self.special_decomposition_fit()
        # self.iterate_parameters()
        self.coefficient = self.__calculate_coefficient__()

    def __calculate_coefficient__(self):
        return (
            self.P0
            * (np.sqrt(2 * np.pi) * self.sigma_rho)
            / np.exp(-(self.m_rho**2) / (2 * self.sigma_rho**2))
        )

    def estimate_P0(self, bandwidth=0.003085):
        """使用核密度估计在 ζ=0 处的概率密度"""
        from scipy.stats import gaussian_kde
        from scipy.stats import iqr
        from scipy.signal import savgol_filter

        data = self.zeta["distance"].values
        n = len(data)
        # 使用Silverman法则计算最优带宽
        silverman_bandwidth = 0.9 * min(np.std(data), iqr(data) / 1.34) * n ** (-0.2)
        print(f"Silverman带宽: {silverman_bandwidth:.6f}")
        # 使用高斯核密度估计
        kde = gaussian_kde(data, bw_method=silverman_bandwidth)
        window_length = 51 if n >= 51 else (n // 2) * 2 + 1  # 确保窗口长度为奇数且不超过数据点数
        polyorder = 3 if window_length > 3 else 2  # 多项式阶数小于窗口长度
        plt.figure(figsize=(10, 6))
        x = np.linspace(data.min(), data.max(), 1000)
        smoothed_kde = savgol_filter(kde(x), window_length=window_length, polyorder=polyorder)
        # plt.plot(x, kde(x), "-k", lw=2, label="KDE Estimate")
        plt.plot(x, smoothed_kde, "--b", lw=2, label="Smoothed KDE")
        hist, bin_edges, _ = plt.hist(
            data, bins=300, density=True, alpha=0.3, color="gray", label="Data Histogram"
        )
        plt.axvline(x=0, color="red", linestyle="--", label="ζ=0")
        plt.xlabel("Zeta Distance (nm)")
        plt.ylabel("Probability Density")
        plt.title("Kernel Density Estimation of Zeta Distance")
        plt.legend()
        # plt.savefig("kde_estimation.png", dpi=300)
        # plt.show()
        P0 = kde(0)[0]
        print(f"Estimated P(0) = {P0:.6f}")
        P0 = smoothed_kde[np.argmin(np.abs(x))]
        print(f"Smoothed Estimated P(0) = {P0:.6f}")
        return x, smoothed_kde, P0

    def special_distribution(self, zeta, m_rho, sigma_rho, m_S, sigma_S):
        """实现图中的特殊分解公式"""
        # 第一项
        exp_factor = np.exp(-(m_rho**2) / (2 * sigma_rho**2))
        term1 = (self.P0 / exp_factor) * np.exp(-((zeta - m_rho) ** 2) / (2 * sigma_rho**2))

        # 第二项的系数
        coefficient = 1 - (sigma_rho * np.sqrt(2 * np.pi) * self.P0) / exp_factor

        # 第二项
        term2 = (
            coefficient
            * np.exp(-((zeta - m_S) ** 2) / (2 * sigma_S**2))
            / (sigma_S * np.sqrt(2 * np.pi))
        )

        return term1 + term2

    def special_decomposition_fit(self):
        """使用特殊分解方式进行拟合"""

        # 初始猜测
        # m_rho 应该接近 0（HDL 均值）
        # m_S 应该大于 0（LDL 均值）
        p0 = [0.0, 0.016, 0.06, 0.03]

        # 参数边界
        # bounds = ([-0.05, 0.001, 0.02, 0.01], [0.05, 0.1, 0.2, 0.1])  # 下界  # 上界

        try:
            # params, cov = curve_fit(
            #     self.special_distribution, bin_centers, hist, p0=p0, bounds=bounds
            # )
            params, cov = curve_fit(
                self.special_distribution, self.x, self.kde, p0=p0, maxfev=100000
            )
            fitted_values = self.special_distribution(self.x, *params)
            r_squared = 1 - np.sum((self.kde - fitted_values) ** 2) / np.sum(
                (self.kde - np.mean(self.kde)) ** 2
            )
            print(f"拟合优度 R² = {r_squared:.6f}")
            return params
        except Exception as e:
            print(f"特殊分解拟合失败: {e}")

    def hdl_distribution(self, x):
        """HDL 分布对应公式的第一项"""
        exp_factor = np.exp(-self.m_rho**2 / (2 * self.sigma_rho**2))
        return (self.P0 / exp_factor) * np.exp(-((x - self.m_rho) ** 2) / (2 * self.sigma_rho**2))

    def ldl_distribution(self, x):
        """LDL 分布对应公式的第二项"""
        exp_factor = np.exp(-self.m_rho**2 / (2 * self.sigma_rho**2))
        coefficient = 1 - (self.sigma_rho * np.sqrt(2 * np.pi) * self.P0) / exp_factor
        return (
            coefficient
            * np.exp(-((x - self.m_S) ** 2) / (2 * self.sigma_S**2))
            / (self.sigma_S * np.sqrt(2 * np.pi))
        )

    def update_parameters(self):
        return self.special_distribution(0, self.m_rho, self.sigma_rho, self.m_S, self.sigma_S)

    def iterate_parameters(self, max_iter=10000, tol=1e-5):
        for i in range(max_iter):
            print(
                f"迭代 {i+1}: m_rho={self.m_rho:.6f}, sigma_rho={self.sigma_rho:.6f}, m_S={self.m_S:.6f}, sigma_S={self.sigma_S:.6f}, P0={self.P0:.6f}"
            )
            current_P0 = self.update_parameters()
            error = self.P0 - current_P0
            print(f"当前 P(0) 估计值: {current_P0:.6f}, 误差: {error:.6e}")
            if abs(error) < tol:
                print(f"参数收敛于迭代 {i}，误差: {error:.6e}")
                break

            # 简单调整 m_rho 和 sigma_rho 以减少误差
            self.P0 = current_P0
            self.m_rho, self.sigma_rho, self.m_S, self.sigma_S = self.special_decomposition_fit()

    def verify_constraint(self):
        """验证 ζ=0 时 LDL 分布为 0 的约束"""
        ldl_at_0 = self.ldl_distribution(0)
        hdl_at_0 = self.hdl_distribution(0)
        total_at_0 = self.update_parameters()

        print("约束验证:")
        print(f"LDL(0) = {ldl_at_0:.10f}")
        print(f"HDL(0) = {hdl_at_0:.10f}")
        print(f"P(0) = {total_at_0:.10f}")
        print(f"估计的 P(0) = {self.P0:.10f}")
        print(f"约束满足: {abs(ldl_at_0) < 1e-10}")

        return abs(ldl_at_0) < 1e-10

    def classify(self, distance):
        """分类函数：计算 HDL 概率"""
        p1 = self.hdl_distribution(distance)
        p2 = self.ldl_distribution(distance)
        # 避免除零错误
        total = p1 + p2
        return p1 / total

    def plot_decomposition(self, path):
        """绘制分解结果"""
        x = np.linspace(self.zeta["distance"].min(), self.zeta["distance"].max(), 1000)

        # 计算各个分量
        total_dist = self.special_distribution(
            x, self.m_rho, self.sigma_rho, self.m_S, self.sigma_S
        )
        hdl_dist = self.hdl_distribution(x)
        ldl_dist = self.ldl_distribution(x)

        # plt.figure(figsize=(12, 8))

        # 绘制直方图
        # hist, bin_edges, _ = plt.hist(
        #     self.zeta["distance"].values,
        #     bins=300,
        #     density=True,
        #     alpha=0.3,
        #     color="gray",
        #     label="Data Histogram",
        # )

        # 绘制分布
        # plt.plot(x, total_dist, "-k", lw=2, label="Total Distribution")
        plt.plot(x, hdl_dist, "--b", lw=2, label=f"HDL Component({self.coefficient:.3f})")
        plt.plot(x, ldl_dist, "--r", lw=2, label="LDL Component")

        # 标记 P=0.5 处对应的 ζ 值
        from scipy.optimize import brentq

        def find_zeta_at_p(target_p):
            return brentq(
                lambda z: self.classify(z) - target_p,
                self.zeta["distance"].min(),
                self.zeta["distance"].max(),
            )

        zeta_at_05 = find_zeta_at_p(0.5)
        plt.axvline(x=zeta_at_05, color="orange", linestyle=":", alpha=0.7, label="P(HDL)=0.5")

        plt.xlabel("Zeta Distance (nm)")
        plt.ylabel("Probability Density")
        plt.title("Two-Gaussian Decomposition of Zeta Distribution")
        plt.legend()
        plt.grid(True, alpha=0.3)
        # plt.yticks(np.arange(0, 15, 6))
        plt.savefig(path, dpi=300)
        # plt.show()

    def plot_classification_probability(self, path):
        """绘制分类概率"""
        x = np.linspace(self.zeta["distance"].min(), self.zeta["distance"].max(), 1000)
        classification_probs = self.classify(x)

        plt.figure(figsize=(10, 6))
        plt.plot(x, classification_probs, "-k", lw=2)

        # 标记 ζ=0 的分类概率
        p_at_0 = self.classify(0)
        plt.axvline(x=0, color="green", linestyle="--", alpha=0.7)
        plt.plot(0, p_at_0, "go", markersize=8, label=f"P(HDL) at ζ=0: {p_at_0:.4f}")

        plt.xlabel("Zeta Distance (nm)")
        plt.ylabel("Probability of HDL Classification")
        plt.title("Classification Probability")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(-0.1, 1.1)
        plt.savefig(path, dpi=300)
        # plt.show()


# 使用示例
zeta_files = [
    "/home/debian/water/TIP4P/2005/nvt/rst/equili/zeta.csv",
    # "/home/debian/water/TIP4P/2005/2020/rst/equili/zeta.csv",
    # "/home/debian/water/TIP4P/2005/Tanaka_2018/rst/zeta.csv",
    # "/home/debian/water/TIP4P/2005/2020/rst/4096/1e-5/zeta.csv",
    # "/home/debian/water/TIP4P/2005/2020/rst/4096/2.5e-5/zeta.csv",
    # "/home/debian/water/TIP4P/2005/2020/rst/4096/7.5e-5/zeta.csv",
    # "/home/debian/water/TIP4P/2005/2020/rst/4096/2.5e-4/zeta.csv",
]

for zeta_file in zeta_files:
    print("=" * 60)
    print("Processing file:", zeta_file)
    print("=" * 60)

    zeta = pd.read_csv(
        zeta_file,
        names=["frame", "O_idx", "distance"],
        header=0,
    )

    classifier = Classifier(zeta)

    # print(f"拟合参数:")
    # print(f"m_ρ = {classifier.m_rho:.6f}")
    # print(f"σ_ρ = {classifier.sigma_rho:.6f}")
    # print(f"m_S = {classifier.m_S:.6f}")
    # print(f"σ_S = {classifier.sigma_S:.6f}")
    # print(f"Coefficient = {classifier.coefficient:.6f}")
    # print()

    path = zeta_file.replace("zeta.csv", "two_gaussian_decomposition.png")
    classifier.plot_decomposition(path)
    path = zeta_file.replace("zeta.csv", "two_gaussian_classification.png")
    classifier.plot_classification_probability(path)
    zeta["hdl_prob"] = zeta["distance"].apply(classifier.classify)
    zeta.to_csv(
        zeta_file.replace(".csv", "_with_classification.csv"),
        index=False,
    )
# classifier.plot_two_gaussian_fit()
# classifier.plot_classification_probability()

# class Classifier:
#     def __init__(self, universe: mda.Universe, hb_count: pd.DataFrame = None):
#         self.universe = universe
#         self.n_atoms = len(universe.atoms)
#         self.O_atoms = self.universe.select_atoms("type 1")
#         self.global_O_indices = self.O_atoms.indices
#         self.hb_count = hb_count
#         # 载入与hb_count对应的帧数据
#         self.start_frame = self.hb_count["frame"].min() if hb_count is not None else 0
#         self.end_frame = (
#             self.hb_count["frame"].max() if hb_count is not None else len(universe.trajectory) - 1
#         )
#         print(f"Classifying frames from {self.start_frame} to {self.end_frame}")
#         self.positions = np.zeros((self.end_frame - self.start_frame + 1, len(self.O_atoms), 3))
#         for ts in tqdm(
#             self.universe.trajectory[self.start_frame : self.end_frame + 1],
#             desc="Loading positions",
#         ):
#             self.positions[ts.frame - self.start_frame] = self.O_atoms.positions.copy()

#     def get_neighbors_within_radius(self, frame: int, radius: float = 5.6):
#         """获取指定帧中每个水分子在给定半径内的邻居索引列表"""
#         frame = frame - self.start_frame  # 调整为positions数组的索引
#         if frame < 0 or frame >= len(self.positions):
#             raise ValueError("帧索引超出范围")
#         coords = self.positions[frame].astype(np.float32)
#         box = self.universe.dimensions.astype(np.float32)  # 获取盒子尺寸

#         searcher = FastNS(radius, coords, box, pbc=True)
#         results = searcher.search(coords)

#         neighbors_list = {i: [] for i in self.global_O_indices}
#         for i, j in zip(results.get_pairs()[:, 0], results.get_pairs()[:, 1]):
#             if i != j:
#                 i = self.global_O_indices[i]
#                 j = self.global_O_indices[j]
#                 neighbors_list[i].append(j)
#         # print(neighbors_list)
#         return neighbors_list

#     def classify_by_hb(self, frame: int):
#         """根据自身氢键数量以及周围第二水合层内的所有水分子的氢键数量对水分子进行分类"""
#         if self.hb_count is None:
#             raise ValueError("氢键计数数据未提供")

#         # 获取当前帧的氢键计数
#         current_hb = self.hb_count[self.hb_count["frame"] == frame]
#         if current_hb.empty:
#             raise ValueError(f"帧 {frame} 的氢键数据不可用")
#         neighbors = self.get_neighbors_within_radius(frame, radius=5.6)
#         for i, row in current_hb.iterrows():
#             hb_self = row["hb_count"]
#             neighbor_indices = neighbors[row["O_idx"]]
#             hb_neighbors = current_hb[current_hb["O_idx"].isin(neighbor_indices)]["hb_count"]
#             if np.all(hb_neighbors == 4) and hb_self == 4:
#                 self.hb_count.at[i, "class"] = "Tetrahedral"
#                 print(f"O_idx {row['O_idx']} classified as Tetrahedral")
#             else:
#                 self.hb_count.at[i, "class"] = "Other"
#         return self.hb_count


# # 示例使用
# u = mda.Universe(
#     "/home/debian/water/TIP4P/2005/2020/4096/traj_2.5e-5_246.lammpstrj", format="LAMMPSDUMP"
# )
# # 假设已经有氢键计数数据
# hb_data = pd.read_csv("/home/debian/water/TIP4P/2005/2020/rst/4096/2.5e-5/hb_counts_per_idx.csv")

# classifier = Classifier(u, hb_count=hb_data)
# classified_data = classifier.classify_by_hb(frame=3700)

# print(classified_data[classified_data["frame"] == 3700])
