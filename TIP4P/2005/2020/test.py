import MDAnalysis as mda
from MDAnalysis.lib.distances import apply_PBC
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata


def cal_disp_mean(pos0, pos):
    """
    pos0: shape(N, 3)
    pos: shape(T, N, 3)
    """
    disp = pos - pos0[np.newaxis, :, :]
    disp_mean = np.mean(disp, axis=0)
    return disp_mean


def plot_2d_displacement(initial_pos, displacement, title="2D Displacement Distribution"):
    # 计算位移大小（模）
    disp_magnitude = np.linalg.norm(displacement, axis=1)

    # 创建3D图形
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)

    # 绘制散点图，颜色表示位移大小
    scatter = ax.scatter(
        initial_pos[:, 0],
        initial_pos[:, 1],
        c=disp_magnitude,
        cmap="hot",
        s=20,
        alpha=0.7,
    )

    # 添加颜色条
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.5)
    cbar.set_label("Displacement Magnitude (Å)", fontsize=12)

    # 设置坐标轴标签
    ax.set_xlabel("X (Å)", fontsize=12)
    ax.set_ylabel("Y (Å)", fontsize=12)

    ax.set_title(title, fontsize=14)

    plt.tight_layout()
    plt.show()

    return fig, ax


def plot_contour_displacement(
    initial_pos,
    displacement,
    title="Displacement Contour Plot",
    n_levels=20,
    interpolation_method="linear",
):
    disp_magnitude = np.linalg.norm(displacement, axis=1)
    x = initial_pos[:, 0]
    y = initial_pos[:, 1]
    z = disp_magnitude

    xi = np.linspace(x.min(), x.max(), 100)
    yi = np.linspace(y.min(), y.max(), 100)
    xi, yi = np.meshgrid(xi, yi)

    zi = griddata((x, y), z, (xi, yi), method=interpolation_method)

    fig, ax = plt.subplots(figsize=(12, 8))

    contourf = ax.contourf(xi, yi, zi, levels=n_levels, cmap="hot", alpha=0.8)
    contour = ax.contour(xi, yi, zi, levels=n_levels, colors="black", linewidths=0.5)

    ax.clabel(contour, inline=True, fontsize=8)
    cbar = plt.colorbar(contourf, ax=ax)
    cbar.set_label("Displacement Magnitude (Å)", fontsize=12)
    ax.set_xlabel("X (Å)", fontsize=12)
    ax.set_ylabel("Y (Å)", fontsize=12)
    ax.set_title(title, fontsize=14)

    plt.tight_layout()
    plt.show()
    return fig, ax


def plot_density_contour(
    initial_pos, displacement, title="Density Contour Plot", bins=50, smoothing=1.0
):
    disp_magnitude = np.linalg.norm(displacement, axis=1)

    x = initial_pos[:, 0]
    y = initial_pos[:, 1]

    fig, ax = plt.subplots(figsize=(12, 8))
    hexbin = ax.hexbin(
        x, y, C=disp_magnitude, gridsize=bins, cmap="hot", reduce_C_function=np.mean, alpha=0.8
    )

    cbar = plt.colorbar(hexbin, ax=ax, shrink=0.5)
    cbar.set_label("Average Displacement Magnitude (Å)", fontsize=12)

    ax.set_xlabel("X (Å)", fontsize=12)
    ax.set_ylabel("Y (Å)", fontsize=12)
    ax.set_title(title, fontsize=14)

    plt.tight_layout()
    plt.show()
    return fig, ax


def plot_3d_displacement(initial_pos, displacement, title="3D Displacement Distribution"):
    """
    绘制3D位移分布图

    Parameters:
    initial_pos: 初始位置，shape (N, 3)
    displacement: 平均位移，shape (N, 3)
    title: 图标题
    """
    # 计算位移大小（模）
    disp_magnitude = np.linalg.norm(displacement, axis=1)

    # 创建3D图形
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    # 绘制散点图，颜色表示位移大小
    scatter = ax.scatter(
        initial_pos[:, 0],
        initial_pos[:, 1],
        initial_pos[:, 2],
        c=disp_magnitude,
        cmap="hot",
        s=20,
        alpha=0.7,
    )

    # 添加颜色条
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.5)
    cbar.set_label("Displacement Magnitude (Å)", fontsize=12)

    # 设置坐标轴标签
    ax.set_xlabel("X (Å)", fontsize=12)
    ax.set_ylabel("Y (Å)", fontsize=12)
    ax.set_zlabel("Z (Å)", fontsize=12)

    ax.set_title(title, fontsize=14)

    # 设置视角
    ax.view_init(elev=20, azim=45)

    plt.tight_layout()
    plt.show()

    return fig, ax


if __name__ == "__main__":
    file_path = "./ice_225.lammpstrj"
    u = mda.Universe(file_path, format="LAMMPSDUMP")
    O_atoms = u.select_atoms("type 1")
    positions = []
    for ts in tqdm(u.trajectory):
        if ts.frame == 0:
            dimensions = u.dimensions
        pos = O_atoms.positions.copy()
        positions.append(pos)
    positions = np.array(positions)
    pos0 = positions[0]
    pos = positions[1:]
    print(pos.shape)
    disp_mean = cal_disp_mean(pos0, pos)
    pos0 = apply_PBC(pos0, dimensions)
    # 画出空间分布图，做一个三维视图
    # 准备数据
    # plot_2d_displacement(pos0, disp_mean)
    plot_contour_displacement(pos0, disp_mean)
    # plot_density_contour(pos0, disp_mean)
    # x = np.linspace(-5, 5, 50)
    # y = np.linspace(-5, 5, 50)
    # X, Y = np.meshgrid(x, y)
    # Z = np.sin(np.sqrt(X**2 + Y**2))

    # # 创建3D热力图
    # fig = plt.figure(figsize=(12, 8))
    # ax = fig.add_subplot(111, projection="3d")
    # surf = ax.plot_surface(X, Y, Z, cmap="coolwarm", linewidth=0, antialiased=False)
    # fig.colorbar(surf, shrink=0.5, aspect=5, label="Z Value")

    # ax.set_title("3D Heatmap - how2matplotlib.com")
    # ax.set_xlabel("X-axis")
    # ax.set_ylabel("Y-axis")
    # ax.set_zlabel("Z-axis")
    # plt.show()

    # disp = positions[1:] - positions[0]
    # disp_mean = np.mean(disp, axis=0)
    # pos_mean = positions[0] + disp_mean
    # # 在三维空间中绘制位置，不用连线，把点画大点
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection="3d")
    # ax.scatter(positions[0, 0], positions[0, 1], positions[0, 2], color="red", s=50)
    # ax.scatter(positions[1:, 0], positions[1:, 1], positions[1:, 2], s=20)
    # ax.scatter(pos_mean[0], pos_mean[1], pos_mean[2], color="blue", s=50)
    # ax.set_xlabel("X (Å)")
    # ax.set_ylabel("Y (Å)")
    # ax.set_zlabel("Z (Å)")
    # ax.set_title("Trajectory of Oxygen Atom 0")
    # plt.show()
