from collections import defaultdict
import MDAnalysis as mda
from MDAnalysis.lib.nsgrid import FastNS
from MDAnalysis.analysis.rdf import InterRDF
from numba import jit, prange
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from MDAnalysis.analysis.hydrogenbonds.hbond_analysis import HydrogenBondAnalysis as HBA
from MDAnalysis.transformations import wrap
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection

# ----------------------------------------------------------------------------------------------------------
# analyze the distribution of zeta value transitions
# ----------------------------------------------------------------------------------------------------------
# zeta_data = pd.read_csv(
#     "/home/debian/water/TIP4P/2005/2020/rst/equili/zeta_with_classification.csv",
#     names=["frame", "O_idx", "distance", "hdl_prob"],
#     header=0,
# )
# # 观察对于每一个O_idx，其hdl_prob发生转变（从小于0.5到大于0.5，从大于0.5到小于0.5）的时间分布
# zeta_data_sorted = zeta_data.sort_values(by=["O_idx", "frame"])
# zeta_transitions = []
# for o_idx, group in zeta_data_sorted.groupby("O_idx"):
#     hdl_probs = group["hdl_prob"].values
#     frames = group["frame"].values
#     for i in range(1, len(hdl_probs)):
#         if (hdl_probs[i - 1] < 0.5 and hdl_probs[i] >= 0.5) or (
#             hdl_probs[i - 1] >= 0.5 and hdl_probs[i] < 0.5
#         ):
#             zeta_transitions.append(
#                 {
#                     "O_idx": o_idx,
#                     "from_frame": frames[i - 1],
#                     "to_frame": frames[i],
#                     "frame": frames[i],
#                 }
#             )
# transitions_df = pd.DataFrame(zeta_transitions)
# # 计算转变时间的分布

# # 计算转变间隔时间（同一个氧原子连续两次转变之间的时间）
# transitions_df_sorted = transitions_df.sort_values(by=["O_idx", "frame"])
# transition_intervals = []

# for o_idx, group in transitions_df_sorted.groupby("O_idx"):
#     frames = group["frame"].values
#     if len(frames) > 1:
#         for i in range(1, len(frames)):
#             interval = frames[i] - frames[i - 1]
#             transition_intervals.append(interval * 0.05)  # 转换为ps

# transition_intervals = np.array(transition_intervals)
# # 计算每个状态（HDL/非HDL）的持续时间
# state_durations = []

# for o_idx, group in zeta_data_sorted.groupby("O_idx"):
#     hdl_probs = group["hdl_prob"].values
#     frames = group["frame"].values

#     current_state = hdl_probs[0] >= 0.5
#     start_frame = frames[0]

#     for i in range(1, len(hdl_probs)):
#         new_state = hdl_probs[i] >= 0.5
#         if new_state != current_state:
#             # 状态转变，记录前一个状态的持续时间
#             duration = (frames[i] - start_frame) * 0.05
#             state_durations.append(duration)
#             current_state = new_state
#             start_frame = frames[i]

# state_durations = np.array(state_durations)
# # 绘制直方图
# plt.figure(figsize=(8, 6))
# plt.hist(state_durations, bins=100, density=True, color="lightgreen", edgecolor="black", alpha=0.7)
# plt.xlabel("State Duration (ps)")
# plt.ylabel("Frequency")
# plt.title("Histogram of Zeta HDL Probability State Durations")
# plt.axvline(
#     np.mean(state_durations),
#     color="black",
#     linestyle="--",
#     label=f"Mean: {np.mean(state_durations):.2f} ps",
# )
# plt.grid(True, alpha=0.3)
# plt.legend()
# plt.tight_layout()
# plt.show()
# # plt.savefig(
# #     "/home/debian/water/TIP4P/2005/2020/rst/4096/2.5e-5/zeta_transition_histogram.png",
# #     dpi=300,
# # )
# plt.show()


# # ----------------------------------------------------------------------------------------------------------
# # test the rearrangement relationship with zeta value
# rearrangement_data = pd.read_hdf(
#     "/home/debian/water/TIP4P/2005/2020/rst/4096/non_trival_displacement_results.h5", key="2.5e-05"
# )
# zeta_data = pd.read_csv(
#     "/home/debian/water/TIP4P/2005/2020/rst/4096/2.5e-5/zeta_with_classification.csv",
#     names=["frame", "O_idx", "distance", "hdl_prob"],
#     header=0,
# )
# # rearrangement_data: columns: O_index, event_frame: event_frame(0-based)
# # zeta_data: columns: frame(2000-based), O_idx, distance, hdl_prob
# frame_offset = 2000
# # find the zeta value for each rearrangement event
# zeta_values = []
# prob_values = []
# cage_times = []
# for o_idx, group in rearrangement_data.groupby("O_index"):
#     event_frames = group["event_frame"].sort_values().values

#     # 计算连续事件之间的时间差（cage time）
#     if len(event_frames) > 1:
#         time_diffs = np.diff(event_frames)
#         cage_times.extend(time_diffs)

#         # 打印一些统计信息（可选）
#         if o_idx < 5:  # 只打印前几个氧原子的信息作为示例
#             print(f"氧原子 {o_idx}: {len(event_frames)} 个事件, {len(time_diffs)} 个笼状时间")
#             print(f"  事件帧: {event_frames}")
#             print(f"  笼状时间: {time_diffs}")

# # 转换为numpy数组便于分析
# cage_times = np.array(cage_times) * 0.05  # 转换为ps，假设时间步长为0.05 ps

# # for _, row in rearrangement_data.iterrows():
# #     O_idx = row["O_index"]
# #     event_frame = row["event_frame"] + frame_offset
# #     zeta_row = zeta_data[(zeta_data["frame"] == event_frame) & (zeta_data["O_idx"] == O_idx)]
# #     if not zeta_row.empty:
# #         zeta_values.append(zeta_row["distance"].values[0])
# #         prob_values.append(zeta_row["hdl_prob"].values[0])
# # plot the histogram of zeta values for rearrangement events
# # plt.figure(figsize=(8, 6))
# # plt.hist(cage_times, bins=100, color="lightcoral", edgecolor="black", alpha=0.7)
# # plt.xlabel("Cage Lifetime (ps)")
# # plt.ylabel("Frequency")
# # plt.axvline(
# #     np.mean(cage_times), color="black", linestyle="--", label=f"Mean: {np.mean(cage_times):.2f} ps"
# # )
# # plt.title("Histogram of Cage Lifetimes for Rearrangement Events")
# # plt.grid(True, alpha=0.3)
# # plt.legend()
# # plt.tight_layout()
# # # plt.savefig(
# # #     "/home/debian/water/TIP4P/2005/2020/rst/4096/2.5e-5/rearrangement_cage_lifetime_histogram.png",
# # #     dpi=300,
# # # )
# # plt.show()
# # plt.figure(figsize=(8, 6))
# # plt.hist(
# #     zeta_values,
# #     bins=100,
# #     density=True,
# #     color="skyblue",
# #     edgecolor="black",
# #     alpha=0.7,
# #     label=r"P(\zeta|R)",
# # )
# # plt.hist(
# #     zeta_data["distance"],
# #     bins=100,
# #     density=True,
# #     color="lightgray",
# #     edgecolor="black",
# #     alpha=0.5,
# #     label=r"P(\zeta)",
# # )
# # plt.xlabel("Zeta Value")
# # plt.ylabel("Frequency")
# # plt.title("Histogram of Zeta Values for Rearrangement Events")
# # plt.grid(True, alpha=0.3)
# # plt.tight_layout()
# # plt.savefig(
# #     "/home/debian/water/TIP4P/2005/2020/rst/4096/2.5e-5/rearrangement_zeta_histogram.png", dpi=300
# # )
# # # plt.show()
# # plt.figure(figsize=(8, 6))
# # plt.hist(prob_values, bins=50, density=True, color="lightgreen", edgecolor="black", alpha=0.7)
# # plt.xlabel("HDL Probability")
# # plt.ylabel("Frequency")
# # plt.title("Histogram of HDL Probabilities for Rearrangement Events")
# # plt.grid(True, alpha=0.3)
# # plt.tight_layout()
# # plt.savefig(
# #     "/home/debian/water/TIP4P/2005/2020/rst/4096/2.5e-5/rearrangement_hdl_prob_histogram.png",
# #     dpi=300,
# # )
# # plt.show()

# -----------------------------------------------------------------------------------------------------------
# Using the HBA class need to be noticed that the idx for atoms are started from 1 in the count_by_ids.
# while in FastNS with indices started from 0 and for results.hbonds the indices are also started from 0.
# 创建HBN(hydrogen Bond Network)分析对象:
file_path = "/home/debian/water/TIP4P/2005/2020/dump_H2O_225.lammpstrj"
hbond_file = "/home/debian/water/TIP4P/2005/2020/rst/equili/test_hbond_counts.csv"
print("\n")
print("=== Hydrogen Bond Analysis ===")
print("\n")

time_step = 0.2  # ps
u = mda.Universe(file_path, format="LAMMPSDUMP")
u.trajectory.add_transformations(wrap(u.atoms))
O_atoms = u.select_atoms("type 1")
O_indices = O_atoms.indices

HBA_analysis = HBA(
    universe=u,
    donors_sel="type 1",
    hydrogens_sel="type 2",
    acceptors_sel="type 1",
    d_a_cutoff=3.5,
    d_h_a_angle_cutoff=180 - 30,
)
start_frame = 5000
stop_frame = 5500
coords = np.zeros((stop_frame - start_frame, len(O_atoms), 3))
# for ts in tqdm(u.trajectory[start_frame:stop_frame]):
#     positions = O_atoms.positions
#     coords[ts.frame - start_frame, :, :] = positions
results = HBA_analysis.run(start=start_frame, stop=stop_frame)
HBA_analysis.run(start=start_frame, stop=stop_frame)
results = HBA_analysis.results.hbonds[:, :-2].astype(int)
# # 保存结果到CSV文件
# pd.DataFrame(results, columns=["frame", "donor_idx", "hydrogen_idx", "acceptor_idx"]).to_csv(
#     hbond_file, index=False
# )
# print(f"氢键分析完成，结果已保存到 {hbond_file}")


# # 去掉最后两列的距离和角度信息，只保留frame, donor_idx, hydrogen_idx, acceptor_idx
# # 构建氢键网络：对于某一帧而言从某一个氧原子出发，递归寻找其所有通过氢键连接的氧原子，直到回到该氧原子或者没有新的氧原子可以加入为止。
# # 对每一帧的每一个氧原子进行上述操作，记录下每一个氧原子所连接的氧原子列表。

# # classification1的方法，如果一个氧原子在某一帧有4个氢键，则记为1，否则记为0，classification2的方法，在某一帧有4个氢键且其相接的这四个氧原子也均有4个氢键，则记为1，否则记为0。
# classification1 = np.zeros((len(O_atoms), stop_frame - start_frame), dtype=int)
# classification2 = np.zeros((len(O_atoms), stop_frame - start_frame), dtype=int)
# for frame in tqdm(range(start_frame, stop_frame)):
#     valid_hbonds = results[results[:, 0] == frame]
#     hbonds_dict = defaultdict(list)
#     for hbond in valid_hbonds:
#         donor_idx = hbond[1]
#         acceptor_idx = hbond[-1]
#         hbonds_dict[donor_idx].append(acceptor_idx)
#         hbonds_dict[acceptor_idx].append(donor_idx)
#     for O_idx in O_indices:
#         local_idx = np.where(O_indices == O_idx)[0][0]
#         if len(hbonds_dict[O_idx]) == 4:
#             classification1[local_idx, frame - start_frame] = 1
#             all_neighbors_fully_coordinated = True
#             for neighbor in hbonds_dict[O_idx]:
#                 if len(hbonds_dict[neighbor]) != 4:
#                     all_neighbors_fully_coordinated = False
#                     break
#             if all_neighbors_fully_coordinated:
#                 classification2[local_idx, frame - start_frame] = 1

# # 保存分类结果
# np.save("/home/debian/water/TIP4P/2005/2020/rst/equili/classification1.npy", classification1)
# np.save("/home/debian/water/TIP4P/2005/2020/rst/equili/classification2.npy", classification2)

# start_frame = 2000
# results = pd.read_csv(
#     "/home/debian/water/TIP4P/2005/2020/rst/equili/test_hbond_counts.csv"
# ).to_numpy()
# classification1 = np.load("/home/debian/water/TIP4P/2005/2020/rst/equili/classification1.npy")
# classification2 = np.load("/home/debian/water/TIP4P/2005/2020/rst/equili/classification2.npy")
# classification3 = pd.read_csv(
#     "/home/debian/water/TIP4P/2005/2020/rst/equili/zeta_with_classification.csv"
# )


def visualize_hbond_network_slice(frame_number, z_center, z_thickness=5.0):
    """
    可视化某一帧中指定z范围slice的氢键网络

    参数:
    frame_number: 要可视化的帧编号（绝对帧号，如2000-3000之间的数字）
    z_center: z轴中心位置(Å)
    z_thickness: z轴厚度(Å)，默认为5Å
    """
    # 计算相对帧号
    rel_frame = frame_number - start_frame

    # 获取该帧的氢键数据
    frame_hbonds = results[results[:, 0] == frame_number]

    # 获取该帧所有氧原子坐标
    frame_coords = coords[rel_frame]
    # frame_classification3 = classification3[classification3["frame"] == frame_number]
    frame_classification2 = classification2[:, rel_frame]

    # 计算z范围
    z_min = z_center - z_thickness / 2
    z_max = z_center + z_thickness / 2

    # 筛选在z范围内的氧原子
    z_mask = (frame_coords[:, 2] >= z_min) & (frame_coords[:, 2] <= z_max)
    slice_O_indices = O_indices[z_mask]
    slice_coords = frame_coords[z_mask]
    # hdl_prob_map = frame_classification3.set_index("O_idx")["hdl_prob"].to_dict()

    # hdl_probs = pd.Series(slice_O_indices).map(hdl_prob_map).values
    hdl_probs = frame_classification2[z_mask]

    print(
        f"帧 {frame_number}: 在z范围 [{z_min:.1f}, {z_max:.1f}] 内有 {len(slice_O_indices)} 个氧原子"
    )

    # 创建图形
    fig = plt.figure(figsize=(16, 8))
    ax1 = fig.add_subplot(111, projection="3d")

    scatter = ax1.scatter3D(
        slice_coords[:, 0],
        slice_coords[:, 1],
        slice_coords[:, 2],
        c=hdl_probs,
        cmap="bwr",
        s=30,
        alpha=0.7,
        label="O原子",
        depthshade=True,
    )

    hbond_count = 0
    hbond_lengths = []
    for hbond in frame_hbonds:
        donor_idx = hbond[1]
        acceptor_idx = hbond[3]

        if donor_idx in slice_O_indices and acceptor_idx in slice_O_indices:
            donor_pos = frame_coords[O_indices == donor_idx][0]
            acceptor_pos = frame_coords[O_indices == acceptor_idx][0]

            direct_distance = np.linalg.norm(donor_pos - acceptor_pos)
            if direct_distance > 3.5:
                continue
            hbond_lengths.append(direct_distance)
            # 绘制氢键连接线
            ax1.plot3D(
                [donor_pos[0], acceptor_pos[0]],
                [donor_pos[1], acceptor_pos[1]],
                [donor_pos[2], acceptor_pos[2]],
                color="red",
                alpha=0.7,
                linewidth=2.5,
            )
            hbond_count += 1
    ax1.set_xlabel("X (Å)")
    ax1.set_ylabel("Y (Å)")
    ax1.set_zlabel("Z (Å)")
    ax1.set_title(
        f"Frame {frame_number}: Hydrogen Bond Network Slice (z={z_center}±{z_thickness/2}Å)"
    )
    cbar = fig.colorbar(scatter, ax=ax1, shrink=0.6, aspect=20)
    cbar.set_label("Z Position (Å)")

    max_range = (
        max(
            [
                slice_coords[:, 0].max() - slice_coords[:, 0].min(),
                slice_coords[:, 1].max() - slice_coords[:, 1].min(),
                slice_coords[:, 2].max() - slice_coords[:, 2].min(),
            ]
        )
        * 0.5
    )

    mid_x = (slice_coords[:, 0].max() + slice_coords[:, 0].min()) * 0.5
    mid_y = (slice_coords[:, 1].max() + slice_coords[:, 1].min()) * 0.5
    mid_z = (slice_coords[:, 2].max() + slice_coords[:, 2].min()) * 0.5

    ax1.set_xlim(mid_x - max_range, mid_x + max_range)
    ax1.set_ylim(mid_y - max_range, mid_y + max_range)
    ax1.set_zlim(mid_z - max_range, mid_z + max_range)

    ax1.view_init(elev=20.0, azim=45)

    # # 子图2: 氢键数量统计
    # hbond_counts = {}
    # for O_idx in slice_O_indices:
    #     # 计算每个氧原子的氢键数量
    #     count = np.sum((frame_hbonds[:, 1] == O_idx) | (frame_hbonds[:, 3] == O_idx))
    #     hbond_counts[O_idx] = count

    # counts = list(hbond_counts.values())

    # if counts:
    #     ax2.hist(
    #         counts, bins=range(0, max(counts) + 2), alpha=0.7, color="skyblue", edgecolor="black"
    #     )
    #     ax2.set_xlabel("number of hydrogen bonds")
    #     ax2.set_ylabel("number of oxygen atoms")
    #     ax2.set_title("Distribution of hydrogen bond numbers")
    #     ax2.grid(True, alpha=0.3)

    #     # 添加统计信息
    #     avg_hbonds = np.mean(counts)
    #     ax2.axvline(avg_hbonds, color="red", linestyle="--", label=f"平均: {avg_hbonds:.2f}")
    #     ax2.legend()

    plt.tight_layout()
    plt.show()

    return slice_coords


def visualize_with_classification(frame_number, z_center, z_thickness=5.0):
    """
    可视化包含分类信息的氢键网络
    """
    rel_frame = frame_number - start_frame

    # 获取该帧的氢键数据
    frame_hbonds = results[results[:, 0] == frame_number]
    frame_coords = coords[rel_frame]

    # 计算z范围
    z_min = z_center - z_thickness / 2
    z_max = z_center + z_thickness / 2

    # 筛选在z范围内的氧原子
    z_mask = (frame_coords[:, 2] >= z_min) & (frame_coords[:, 2] <= z_max)
    slice_O_indices = O_indices[z_mask]
    slice_coords = frame_coords[z_mask]

    # 获取分类信息
    class1_mask = classification1[z_mask, rel_frame] == 1
    class2_mask = classification2[z_mask, rel_frame] == 1

    fig, ax = plt.subplots(figsize=(10, 8))

    # 绘制不同分类的氧原子
    # 普通氧原子（非4配位）
    normal_mask = ~class1_mask
    ax.scatter(
        slice_coords[normal_mask, 0],
        slice_coords[normal_mask, 1],
        c="lightblue",
        s=50,
        alpha=0.7,
        label="普通氧原子",
    )

    # 4配位氧原子（classification1）
    class1_pos = slice_coords[class1_mask & ~class2_mask]
    ax.scatter(
        class1_pos[:, 0],
        class1_pos[:, 1],
        c="orange",
        s=80,
        alpha=0.8,
        label="4配位氧原子",
        marker="s",
    )

    # 完美4配位氧原子（classification2）
    class2_pos = slice_coords[class2_mask]
    ax.scatter(
        class2_pos[:, 0], class2_pos[:, 1], c="red", s=100, alpha=1.0, label="完美4配位", marker="*"
    )

    # 绘制氢键连接
    hbond_lines = []
    for hbond in frame_hbonds:
        donor_idx = hbond[1]
        acceptor_idx = hbond[3]

        if donor_idx in slice_O_indices and acceptor_idx in slice_O_indices:
            donor_pos = frame_coords[O_indices == donor_idx][0]
            acceptor_pos = frame_coords[O_indices == acceptor_idx][0]

            hbond_lines.append([(donor_pos[0], donor_pos[1]), (acceptor_pos[0], acceptor_pos[1])])

    if hbond_lines:
        lc = LineCollection(hbond_lines, colors="green", alpha=0.5, linewidths=1)
        ax.add_collection(lc)

    ax.set_xlabel("X (Å)")
    ax.set_ylabel("Y (Å)")
    ax.set_title(
        f"帧 {frame_number}: 氢键网络与分类 (z={z_center}±{z_thickness/2}Å)\n"
        f"4配位: {np.sum(class1_mask)}个, 完美4配位: {np.sum(class2_mask)}个"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")

    plt.tight_layout()
    plt.show()


# save classification results
# np.save("classification1.npy", classification1)
# np.save("classification2.npy", classification2)
# plot the classification results
# classification1 = np.load("classification1.npy")
# classification2 = np.load("classification2.npy")
# print("Classification 1 shape:", classification1.shape)
# print("Classification 2 shape:", classification2.shape)


# 创建动画函数
def create_spatial_animation(
    start_frame,
    stop_frame,
    coords,
    classification1,
    classification2,
    z_range=(14.5, 19.5),
    save_animation=True,
    filename="hydrogen_bond_animation.gif",
):
    """
    创建氢键分类空间分布的动画

    参数:
    - start_frame, stop_frame: 帧范围
    - coords: 坐标数组
    - classification1, classification2: 分类结果
    - z_range: z轴范围筛选
    - save_animation: 是否保存动画文件
    - filename: 保存的文件名
    """

    # 创建图形
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle("Spatial Distribution of Hydrogen Bond Classification Over Time", fontsize=16)

    # 设置坐标轴范围
    x_min, x_max = coords[:, :, 0].min(), coords[:, :, 0].max()
    y_min, y_max = coords[:, :, 1].min(), coords[:, :, 1].max()

    # 初始化散点图
    def get_frame_data(frame_idx):
        """获取指定帧的数据"""
        frame_relative = frame_idx - start_frame
        valid_mask = (z_range[0] < coords[frame_relative, :, 2]) & (
            coords[frame_relative, :, 2] < z_range[1]
        )

        x_coords = coords[frame_relative, :, 0][valid_mask]
        y_coords = coords[frame_relative, :, 1][valid_mask]
        class1_vals = classification1[:, frame_relative][valid_mask]
        class2_vals = classification2[:, frame_relative][valid_mask]

        return x_coords, y_coords, class1_vals, class2_vals

    # 第一帧数据
    x_coords, y_coords, class1_vals, class2_vals = get_frame_data(start_frame)

    # 创建散点图
    sc1 = ax1.scatter(x_coords, y_coords, c=class1_vals, cmap="viridis", s=20, vmin=0, vmax=1)
    sc2 = ax2.scatter(x_coords, y_coords, c=class2_vals, cmap="viridis", s=20, vmin=0, vmax=1)

    # 设置图形属性
    ax1.set_xlim(x_min, x_max)
    ax1.set_ylim(y_min, y_max)
    ax2.set_xlim(x_min, x_max)
    ax2.set_ylim(y_min, y_max)

    ax1.set_xlabel("X Position (Å)")
    ax1.set_ylabel("Y Position (Å)")
    ax1.set_title("Classification 1: 4 Hydrogen Bonds")

    ax2.set_xlabel("X Position (Å)")
    ax2.set_ylabel("Y Position (Å)")
    ax2.set_title("Classification 2: Fully Coordinated Neighborhood")

    # 添加颜色条
    cbar1 = plt.colorbar(sc1, ax=ax1)
    cbar1.set_label("Classification Value")
    cbar2 = plt.colorbar(sc2, ax=ax2)
    cbar2.set_label("Classification Value")

    # 添加帧数文本
    frame_text = fig.text(
        0.5,
        0.02,
        f"Frame: {start_frame}",
        ha="center",
        fontsize=12,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )

    # 统计文本
    stats_text = fig.text(
        0.5,
        0.95,
        "",
        ha="center",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )

    def update(frame_idx):
        """更新动画帧"""
        frame_relative = frame_idx - start_frame
        x_coords, y_coords, class1_vals, class2_vals = get_frame_data(frame_idx)

        # 更新散点图数据
        sc1.set_offsets(np.column_stack([x_coords, y_coords]))
        sc1.set_array(class1_vals)

        sc2.set_offsets(np.column_stack([x_coords, y_coords]))
        sc2.set_array(class2_vals)

        # 更新帧数文本
        frame_text.set_text(f"Frame: {frame_idx}")

        # 更新统计信息
        class1_count = np.sum(class1_vals)
        class2_count = np.sum(class2_vals)
        total_atoms = len(class1_vals)
        stats_text.set_text(
            f"Class 1: {class1_count}/{total_atoms} ({class1_count/total_atoms*100:.1f}%) | "
            f"Class 2: {class2_count}/{total_atoms} ({class2_count/total_atoms*100:.1f}%)"
        )

        return sc1, sc2, frame_text, stats_text

    # 创建动画
    anim = FuncAnimation(
        fig, update, frames=range(start_frame, stop_frame), interval=100, blit=False, repeat=True
    )

    # 保存动画（可选）
    if save_animation:
        print("Saving animation...")
        anim.save(filename, writer="pillow", fps=10, dpi=100)
        print(f"Animation saved as {filename}")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    return anim


# 创建动画
# anim = create_spatial_animation(
#     start_frame,
#     stop_frame,
#     coords,
#     classification1,
#     classification2,
#     z_range=(14.5, 19.5),
#     save_animation=True,
# )


# 可选：创建简化的动画（更少的帧，用于快速测试）
def create_fast_animation(
    start_frame,
    stop_frame,
    coords,
    classification1,
    classification2,
    frame_step=10,
    z_range=(14.5, 19.5),
):
    """创建快速动画（跳帧）用于测试"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # 获取第一帧数据
    frame_relative = 0
    valid_mask = (z_range[0] < coords[frame_relative, :, 2]) & (
        coords[frame_relative, :, 2] < z_range[1]
    )
    x_coords = coords[frame_relative, :, 0][valid_mask]
    y_coords = coords[frame_relative, :, 1][valid_mask]
    class1_vals = classification1[:, frame_relative][valid_mask]
    class2_vals = classification2[:, frame_relative][valid_mask]

    sc1 = ax1.scatter(x_coords, y_coords, c=class1_vals, cmap="viridis", s=20)
    sc2 = ax2.scatter(x_coords, y_coords, c=class2_vals, cmap="viridis", s=20)

    ax1.set_title("Classification 1")
    ax2.set_title("Classification 2")

    def update_fast(frame_idx):
        frame_relative = frame_idx - start_frame
        valid_mask = (z_range[0] < coords[frame_relative, :, 2]) & (
            coords[frame_relative, :, 2] < z_range[1]
        )
        x_coords = coords[frame_relative, :, 0][valid_mask]
        y_coords = coords[frame_relative, :, 1][valid_mask]
        class1_vals = classification1[:, frame_relative][valid_mask]
        class2_vals = classification2[:, frame_relative][valid_mask]

        sc1.set_offsets(np.column_stack([x_coords, y_coords]))
        sc1.set_array(class1_vals)
        sc2.set_offsets(np.column_stack([x_coords, y_coords]))
        sc2.set_array(class2_vals)

        ax1.set_title(f"Classification 1 - Frame {frame_idx}")
        ax2.set_title(f"Classification 2 - Frame {frame_idx}")

        return sc1, sc2

    frames_to_show = range(start_frame, stop_frame, frame_step)
    anim_fast = FuncAnimation(fig, update_fast, frames=frames_to_show, interval=200, blit=False)

    plt.tight_layout()
    plt.show()

    return anim_fast


# # 可视化某一帧的氢键网络切片
# visualize_hbond_network_slice(frame_number=6000, z_center=20, z_thickness=5.0)
# # 如果需要快速测试，取消注释下面这行
# # anim_fast = create_fast_animation(
# #     start_frame, stop_frame, coords, classification1, classification2, frame_step=50
# # )
# time_target = 0.61647680  # ps
# frame_region = int(time_target / time_step)  # 80 frames
# print(f"Frame region for {time_target} ps: {frame_region}")
# # frame_region = 80
# # 检查在80帧内，classification的变化次数情况
# changes_counts1 = []
# changes_counts2 = []
# for i in range(len(O_atoms)):
#     changes1 = 0
#     changes2 = 0
#     for frame in range(start_frame, start_frame + frame_region - 1):
#         if classification1[i, frame - start_frame] != classification1[i, frame + 1 - start_frame]:
#             changes1 += 1
#         if classification2[i, frame - start_frame] != classification2[i, frame + 1 - start_frame]:
#             changes2 += 1
#     changes_counts1.append(changes1)
#     changes_counts2.append(changes2)
# plt.figure()
# plt.hist(
#     changes_counts1,
#     bins=range(0, max(changes_counts1) + 2),
#     alpha=0.5,
#     label="Classification 1",
# )
# plt.hist(
#     changes_counts2,
#     bins=range(0, max(changes_counts2) + 2),
#     alpha=0.5,
#     label="Classification 2",
# )
# plt.xlabel("Number of Changes in Classification (over 80 frames)")
# plt.ylabel("Frequency")
# plt.title("Distribution of Classification Changes Over 80 Frames")
# plt.legend()
# plt.grid(True)
# plt.show()

# lifetime = results.lifetime(tau_max=60, window_step=15)
# plt.figure()
# plt.plot(lifetime[0] * time_step, lifetime[1], marker="o")
# plt.xlabel("Time (ps)")
# plt.ylabel("Number of Hydrogen Bonds")
# plt.title("Hydrogen Bond Lifetime")
# plt.grid(True)
# plt.show()


# u.trajectory[stop_frame]

# ns = FastNS(cutoff=4.0, coords=O_atoms.positions, box=u.dimensions, pbc=True)
# results = ns.self_search()
# neighbor_dict = defaultdict(list)
# for distance, pair in zip(results.get_pair_distances(), results.get_pairs()):
#     idx1, idx2 = pair
#     O_idx1 = O_indices[idx1]
#     O_idx2 = O_indices[idx2]
#     neighbor_dict[O_idx1].append((O_idx2, distance))
#     neighbor_dict[O_idx2].append((O_idx1, distance))
# # 计算every O atom 满足上述条件的邻居数目
# print(neighbor_dict)
# print(neighbor_dict.keys())
# counts = []
# counts_strict = []
# counts_more_strict = []
# print(f"len neighbor_dict keys: {len(neighbor_dict.keys())}")
# for O_idx in neighbor_dict.keys():
#     neighbors = neighbor_dict[O_idx]
#     count = len(neighbors)
#     count_strict = 0
#     count_more_strict = 0
#     for neighbor in neighbors:
#         if neighbor[1] < 3.5:  # 对应氢键距离截断
#             count_strict += 1
#         if neighbor[1] < 3.2:  # 对应第一层壳峰谷值
#             count_more_strict += 1
#     counts.append(count)
#     counts_strict.append(count_strict)
#     counts_more_strict.append(count_more_strict)
# # 绘制邻居数目的分布直方图
# plt.figure()
# plt.hist(counts, bins=range(0, max(counts) + 2), alpha=0.5, label="All Neighbors (<4.0 Å)")
# plt.hist(
#     counts_strict,
#     bins=range(0, max(counts_strict) + 2),
#     alpha=0.5,
#     label="Strict Neighbors (<3.5 Å)",
# )
# plt.hist(
#     counts_more_strict,
#     bins=range(0, max(counts_more_strict) + 2),
#     alpha=0.5,
#     label="More Strict Neighbors (<3.2 Å)",
# )
# plt.xlabel("Number of Neighbors")
# plt.ylabel("Frequency")
# plt.title("Distribution of Neighbors for Oxygens")
# plt.legend()
# plt.grid(True)
# plt.show()


# hbond_data = pd.read_csv(hbond_file)
# fraction_list = []
# for frame in tqdm(range(start_frame, stop_frame)):
#     valid_hbonds = hbond_data[hbond_data["frame"] == frame]
#     u.trajectory[frame]
#     ns = FastNS(cutoff=5.5, coords=O_atoms.positions, box=u.dimensions, pbc=True)
#     results = ns.self_search()
#     neighbor_dict = defaultdict(set)
#     count_in_frame = 0
#     for pair in results.get_pairs():
#         idx1, idx2 = pair
#         O_idx1 = O_indices[idx1]
#         O_idx2 = O_indices[idx2]
#         neighbor_dict[O_idx1].add(O_idx2)
#         neighbor_dict[O_idx2].add(O_idx1)
#     for key in neighbor_dict.keys():
#         neighbors = neighbor_dict[key]
#         if np.all(valid_hbonds[valid_hbonds["water_idx"] == key] == 4) and np.all(
#             valid_hbonds[valid_hbonds["water_idx"].isin(neighbors)] == 4
#         ):
#             count_in_frame += 1
#     fraction = count_in_frame / len(O_atoms)
#     fraction_list.append(fraction)
# plt.figure()
# plt.plot(range(start_frame, stop_frame), fraction_list, marker="o")
# plt.xlabel("Frame")
# plt.ylabel("Fraction of Fully Coordinated Oxygens")
# plt.title("Fraction of Fully Coordinated Oxygens vs Frame")
# plt.grid(True)
# plt.show()


tau_max = int(3 / 0.05)  # 最大寿命，单位为帧数
time, tau = HBA_analysis.lifetime(tau_max=tau_max)
plt.figure()
plot = plt.plot(time * 0.2, tau, marker="o")
plt.yscale("log")
plt.xlabel("Time (ps)")
plt.ylabel("Number of Hydrogen Bonds")
plt.title("Hydrogen Bond Lifetime")
plt.grid(True)
plt.show()
