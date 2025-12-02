import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm

color_map = {"HDL": "blue", "LDL": "red"}


def apply_shear_correction(coords, shear_rate, time_step):
    n_frames = coords.shape[0]
    for frame in tqdm(range(n_frames), desc="Applying shear correction"):
        if frame == 0:
            continue
        y_positions = coords[frame - 1, :, 1]
        # 修正x坐标以消除剪切流影响
        coords[frame:, :, 0] -= shear_rate * time_step * y_positions
    return coords


def cal_max_displacement_during_time_interval(coords, delta_t):
    """计算在给定时间间隔内的最大位移及其对应的时间起点"""
    n_frames = coords.shape[0]
    n_particles = coords.shape[1]

    disp_records = []
    for t0 in range(n_frames - delta_t):
        max_disp = np.zeros(n_particles)
        max_t = np.zeros(n_particles)
        for t in range(delta_t):
            disp = np.linalg.norm(coords[t0 + t] - coords[t0], axis=1)
            max_t = np.where(disp > max_disp, t, max_t)
            np.maximum(max_disp, disp, out=max_disp)
        disp_records.append((max_disp, max_t))

    return disp_records


def plot_max_displacement_over_time(coords, delta_t):
    disp_records = cal_max_displacement_during_time_interval(coords, delta_t)
    max_t = [record[1] for record in disp_records]
    max_disps = [record[0] for record in disp_records]
    # times = np.arange(len(max_disps))

    plt.figure(figsize=(8, 6))
    # plt.plot(times, max_disps, marker="o")
    # plt.hist(np.concatenate(max_t), bins=30)
    plt.hist(
        np.log(np.concatenate(max_disps)),
        bins=100,
        density=True,
        color="skyblue",
        edgecolor="black",
        alpha=0.7,
    )
    plt.xlabel("Max Displacement")
    plt.xscale("log")
    plt.ylabel("Frequency")
    plt.yscale("log")
    plt.title(f"Max Displacement over Time (Δt={delta_t} frames)")
    plt.grid()
    plt.figure(figsize=(8, 6))
    plt.hist(
        np.concatenate(max_t),
        bins=30,
        density=True,
        color="lightgreen",
        edgecolor="black",
        alpha=0.7,
    )
    plt.xlabel("Time to Max Displacement (frames)")
    plt.ylabel("Frequency")
    plt.title(f"Time to Max Displacement Distribution (Δt={delta_t} frames)")
    plt.grid()
    plt.show()


def assign_colors_by_type(particle_types, frame):
    colors = []
    for particle_id in particle_types:
        current_type = particle_types[particle_id, frame]
        colors.append(color_map[current_type])
    return colors


def plot_trajectories_with_type_changes(coords, particle_types, time_window=100):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")

    n_particles = coords.shape[1]

    for particle_id in range(n_particles):
        traj = coords[:time_window, particle_id, :]
        type_seq = particle_types[particle_id, :time_window]

        change_points = np.where(type_seq[:-1] != type_seq[1:])[0] + 1
        segments = np.split(np.arange(len(traj)), change_points)

        for i, seg in enumerate(segments):
            if len(seg) > 1:
                seg_traj = traj[seg]
                seg_type = type_seq[seg[0]]
                ax.plot(
                    seg_traj[:, 0],
                    seg_traj[:, 1],
                    seg_traj[:, 2],
                    color=color_map[seg_type],
                    alpha=0.6,
                    linewidth=1,
                )

        ax.scatter(*traj[0], color="green", s=20, marker="o")
        ax.scatter(*traj[-1], color="black", s=20, marker="s")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.title("Particle Trajectories with Type Changes")
    plt.show()


def plot_local_concentration(coordinates, particle_types, frame, grid_size=3):
    """计算并显示局部HDL/LDL浓度分布"""
    from scipy import stats

    positions = coordinates[frame]
    types = [particle_types[i, frame] for i in range(len(positions))]

    # 创建网格
    x_min, x_max = positions[:, 0].min(), positions[:, 0].max()
    y_min, y_max = positions[:, 1].min(), positions[:, 1].max()

    xi = np.linspace(x_min, x_max, grid_size)
    yi = np.linspace(y_min, y_max, grid_size)

    # 计算每个网格点的HDL浓度
    hdl_positions = positions[[i for i, t in enumerate(types) if t == "HDL"]]

    if len(hdl_positions) > 0:
        # 使用KDE估计密度
        kde = stats.gaussian_kde(hdl_positions[:, :2].T)
        X, Y = np.meshgrid(xi, yi)
        Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)

        plt.figure(figsize=(10, 8))
        plt.contourf(X, Y, Z, levels=20, cmap="Blues")
        plt.colorbar(label="HDL Local Concentration")

        # 叠加粒子位置
        for i, (pos, p_type) in enumerate(zip(positions, types)):
            color = color_map[p_type]
            plt.scatter(pos[0], pos[1], c=color, s=50, edgecolors="black", linewidth=0.5)

        plt.title(f"Local Concentration at Frame {frame}")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.show()


def analyze_type_clustering(coordinates, particle_types, frame):
    """分析类型聚集情况"""

    positions = coordinates[frame]
    types = [particle_types[i, frame] for i in range(len(positions))]

    plt.figure(figsize=(12, 6))

    # 子图2：显示类型分布
    for i, (pos, p_type) in enumerate(zip(positions, types)):
        color = color_map[p_type]
        plt.scatter(pos[0], pos[1], c=color, s=50)
    plt.title("Type Distribution")
    plt.xlabel("X")
    plt.ylabel("Y")

    plt.tight_layout()
    plt.show()

    # # 计算团簇的类型纯度
    # cluster_purities = []
    # for cluster_id in unique_clusters:
    #     cluster_indices = np.where(labels == cluster_id)[0]
    #     cluster_types = [types[i] for i in cluster_indices]
    #     hdl_fraction = cluster_types.count("HDL") / len(cluster_types)
    #     cluster_purities.append(hdl_fraction)

    # return cluster_purities


def plot_displacement_vecmap(coordinates, particle_types, time_interval=10):
    """绘制位移的矢量图"""
    n_frames = coordinates.shape[0]
    n_particles = coordinates.shape[1]

    fig, ax = plt.subplots(figsize=(10, 8))
    disp_vectors = []
    for particle_id in range(n_particles):
        start_pos = coordinates[0, particle_id, :2]
        end_pos = coordinates[time_interval, particle_id, :2]
        disp_vec = end_pos - start_pos
        disp_vectors.append(disp_vec)

    disp_vectors = np.array(disp_vectors)
    X = coordinates[0, :, 0]
    Y = coordinates[0, :, 1]
    X, Y = np.meshgrid(X, Y)
    ax.quiver(
        coordinates[0, :, 0],
        coordinates[0, :, 1],
        disp_vectors[:, 0],
        disp_vectors[:, 1],
        angles="xy",
        scale_units="xy",
        scale=1,
        color="teal",
        alpha=0.7,
    )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(f"Displacement Vector Map over {time_interval} Frames")
    plt.grid()
    plt.show()


def plot_trajectory_for_one_particle(coords, particle_id, particle_types):
    """绘制单个粒子的三维轨迹,并根据粒子类型变化进行颜色区分"""
    traj = coords[:, particle_id, :]
    types = particle_types[particle_id, :]
    fig = plt.figure(figsize=(10, 8))
    # 根据类型变化分段绘制轨迹
    change_points = np.where(types[:-1] != types[1:])[0] + 1
    segments = np.split(np.arange(len(traj)), change_points)
    ax = fig.add_subplot(111, projection="3d")
    for i, seg in enumerate(segments):
        if len(seg) > 1:
            seg_traj = traj[seg]
            seg_type = types[seg[0]]
            ax.plot(
                seg_traj[:, 0],
                seg_traj[:, 1],
                seg_traj[:, 2],
                color=color_map[seg_type],
                alpha=0.7,
                linewidth=2,
            )
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.title(f"Trajectory of Particle {particle_id} with Type Changes")
    plt.legend()
    plt.show()
    # ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], color="purple", alpha=0.7)
    # ax.scatter(*traj[0], color="green", s=50, label="Start")
    # ax.scatter(*traj[-1], color="red", s=50, label="End")

    # ax.set_xlabel("X")
    # ax.set_ylabel("Y")
    # ax.set_zlabel("Z")
    # plt.title(f"Trajectory of Particle {particle_id}")
    # plt.legend()
    # plt.show()


if __name__ == "__main__":
    # df = pd.read_csv("/home/debian/water/TIP4P/2005/2020/rst/equili/zeta_cg_L1.csv")
    # 跟踪展示每个粒子在时间尺度上发生了多少次的分类变化
    # 分成多个时间段来统计每个时间段内的分类变化次数
    # frames = df["frame"].unique() - 9500
    # hdl_prob = df["distance"].values
    # classification = np.where(hdl_prob >= 0.045, "LDL", "HDL")
    # O_ids = df["O_idx"].unique() // 3
    traj_file = "/home/debian/water/TIP4P/2005/2020/dump_H2O_225.lammpstrj"
    import MDAnalysis as mda
    from MDAnalysis.transformations import wrap

    u = mda.Universe(traj_file, format="LAMMPSDUMP")
    # u.trajectory.add_transformations(wrap(u.atoms))
    start_frame = 9000
    O_atoms = u.select_atoms("type 1")
    frames = np.arange(0, 1000)  # 分析1000帧
    O_ids = O_atoms.indices
    coords = np.zeros((len(frames), len(O_ids), 3))
    for ts in tqdm(
        u.trajectory[start_frame : start_frame + len(frames)], desc="Extracting coordinates"
    ):
        current_frame = ts.frame - start_frame
        coords[current_frame] = O_atoms.positions.copy()
    print("Coordinates extracted.")

    # shear_rate = 2.5e-2  # 1/ps
    time_step = 0.2  # ps

    # 启用函数进行剪切校正
    # coords = apply_shear_correction(coords, shear_rate, time_step)

    # plot_local_concentration(coords, classification.reshape(len(O_ids), len(frames)), frame=0)
    # analyze_type_clustering(coords, classification.reshape(len(O_ids), len(frames)), frame=200)

    # 绘制最大动态不均匀性时间间隔下的位移热图
    time_x = 68  # ps
    # time_x = 0.61647680  # ps
    time_step = 0.2  # ps
    time_interval = int(time_x / time_step)
    # plot_trajectory_for_one_particle(
    #     coords, particle_id=66, particle_types=classification.reshape(len(O_ids), len(frames))
    # )
    plot_max_displacement_over_time(coords, time_interval)
    # plot_displacement_vecmap(
    #     coords, classification.reshape(len(O_ids), len(frames)), time_interval=time_interval
    # )
    # plot_trajectories_with_type_changes(
    #     coords, classification.reshape(len(O_ids), len(frames)), time_window=100
    # )

    # # 统计每个粒子在每个time_x下的分类变化的次数并绘制直方图
    # change_time_counts = {}
    # change_from_HDL_to_LDL = {}
    # change_from_LDL_to_HDL = {}
    # for time in range(len(frames) - time_interval):
    #     for O_id in O_ids:
    #         class_series = classification.reshape(len(O_ids), len(frames))[O_id]
    #         changes = np.sum(
    #             class_series[time + 1 : time + time_interval]
    #             != class_series[time : time + time_interval - 1]
    #         )
    #         change_from_HDL_to_LDL[(time, O_id)] = np.sum(
    #             (class_series[time + 1 : time + time_interval] == "LDL")
    #             & (class_series[time : time + time_interval - 1] == "HDL")
    #         )
    #         change_from_LDL_to_HDL[(time, O_id)] = np.sum(
    #             (class_series[time + 1 : time + time_interval] == "HDL")
    #             & (class_series[time : time + time_interval - 1] == "LDL")
    #         )
    #         change_time_counts[(time, O_id)] = changes
    #         change_from_HDL_to_LDL[(time, O_id)] = change_from_HDL_to_LDL.get((time, O_id), 0)
    #         change_from_LDL_to_HDL[(time, O_id)] = change_from_LDL_to_HDL.get((time, O_id), 0)
    # change_time_counts_series = pd.Series(change_time_counts)
    # change_from_HDL_to_LDL_series = pd.Series(change_from_HDL_to_LDL)
    # change_from_LDL_to_HDL_series = pd.Series(change_from_LDL_to_HDL)
    # plt.figure(figsize=(10, 6))
    # plt.hist(change_time_counts_series, bins=30, density=True, color="skyblue", edgecolor="black")
    # plt.title(f"Distribution of Classification Changes per Particle (time_x={time_x} ps)")
    # plt.xlabel("Number of Classification Changes")
    # plt.ylabel("Number of Particles")
    # plt.grid(axis="y", alpha=0.75)
    # plt.show()
    # plt.figure(figsize=(10, 6))
    # plt.hist(
    #     change_from_HDL_to_LDL_series, bins=30, density=True, color="orange", edgecolor="black"
    # )
    # plt.title(f"Distribution of HDL to LDL Changes per Particle (time_x={time_x} ps)")
    # plt.xlabel("Number of HDL to LDL Changes")
    # plt.ylabel("Number of Particles")
    # plt.grid(axis="y", alpha=0.75)
    # plt.show()
    # plt.figure(figsize=(10, 6))
    # plt.hist(change_from_LDL_to_HDL_series, bins=30, density=True, color="green", edgecolor="black")
    # plt.title(f"Distribution of LDL to HDL Changes per Particle (time_x={time_x} ps)")
    # plt.xlabel("Number of LDL to HDL Changes")
    # plt.ylabel("Number of Particles")
    # plt.grid(axis="y", alpha=0.75)
    # plt.show()

    # for O_id in O_ids:
    #     class_series = classification[df["O_idx"] == O_id]
    #     changes = np.sum(class_series[1:] != class_series[:-1])
    #     change_counts[O_id] = changes
    # change_counts_series = pd.Series(change_counts)
    # plt.figure(figsize=(10, 6))
    # plt.hist(change_counts_series, bins=30, color="skyblue", edgecolor="black")
    # plt.title("Distribution of Classification Changes per Particle")
    # plt.xlabel("Number of Classification Changes")
    # plt.ylabel("Number of Particles")
    # plt.grid(axis="y", alpha=0.75)
    # # plt.savefig("/home/debian/water/TIP4P/2005/2020/rst/4096/2.5e-5/classification_changes_histogram.png")
    # plt.show()
