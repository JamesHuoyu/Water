"""
计算 non-trival displacement 的工具
"""

import numpy as np
import MDAnalysis as mda
from tqdm import tqdm
import pandas as pd
import h5py


# 计算对于单粒子的 non-trival displacement
def apply_shear_correction(coords, shear_rate, time_step):
    n_frames = coords.shape[0]
    for frame in tqdm(range(n_frames), desc="Applying shear correction"):
        if frame == 0:
            continue
        y_positions = coords[frame - 1, :, 1]
        # 修正x坐标以消除剪切流影响
        coords[frame:, :, 0] -= shear_rate * time_step * y_positions
    return coords


def compute_time_averaged(positions):
    """
    计算单粒子的 time-averaged position

    参数:
    positions: np.ndarray, 形状为 (T, 3)，表示粒子在 T 个时间步的三维位置

    返回:
    time_averaged_positions: np.ndarray, 形状为 (3,)，表示时间平均位置
    """
    return np.mean(positions, axis=0)


def compute_non_trival_displacement(positions, time_interval):
    """
    计算单粒子的 non-trival displacement

    参数:
    positions: np.ndarray, 形状为 (T, 3)，表示粒子在 T 个时间步的三维位置
    time_interval: int, 时间间隔，用于计算位移

    返回:
    p_hop_list: np.ndarray, 形状为 (T,)，表示每个时间步的 non-trival displacement
    """
    time_interpret = time_interval // 2
    if time_interpret >= positions.shape[0] // 2:
        raise ValueError("时间间隔过大，无法计算非平凡位移")
    pos_avg_b = np.zeros(shape=(positions.shape[0], 3))
    pos_avg_a = np.zeros(shape=(positions.shape[0], 3))
    for t in range(positions.shape[0]):
        valid_end_b = min(t + time_interpret, positions.shape[0])
        valid_start_a = max(t - time_interpret, 0)
        pos_avg_b[t] = compute_time_averaged(positions[t:valid_end_b])
        pos_avg_a[t] = compute_time_averaged(positions[valid_start_a:t])
    pos_valid_b_sq = np.linalg.norm(positions - pos_avg_b, axis=1) ** 2  # shape (T,)
    pos_valid_a_sq = np.linalg.norm(positions - pos_avg_a, axis=1) ** 2  # shape (T,)
    p_hop_list = np.zeros(shape=(positions.shape[0],))
    for t in range(positions.shape[0]):
        valid_end_b = min(t + time_interpret, positions.shape[0])
        valid_start_a = max(t - time_interpret, 0)
        sq_a = np.mean(pos_valid_b_sq[valid_start_a:t], axis=0)
        sq_b = np.mean(pos_valid_a_sq[t:valid_end_b], axis=0)
        p_hop = np.sqrt(sq_a * sq_b)  # shape ()
        p_hop_list[t] = p_hop
    mask = p_hop_list > 0
    return p_hop_list


def compute_non_trival_disp(positions):
    """
    计算单粒子的 non-trival displacement，不使用时间间隔参数,而是使用整个时间序列

    参数:
    positions: np.ndarray, 形状为 (T, 3)，表示粒子在 T 个时间步的三维位置

    返回:
    p_hop_list: np.ndarray, 形状为 (T,)，表示每个时间步的 non-trival displacement
    """
    T = positions.shape[0]
    pos_avg_b = np.zeros(shape=(T, 3))
    pos_avg_a = np.zeros(shape=(T, 3))
    for t in range(T):
        b_interval = slice(t, T)
        a_interval = slice(0, t + 1)
        pos_avg_b[t] = compute_time_averaged(positions[b_interval])
        pos_avg_a[t] = compute_time_averaged(positions[a_interval])
    pos_valid_b_sq = np.linalg.norm(positions - pos_avg_b, axis=1) ** 2  # shape (T,)
    pos_valid_a_sq = np.linalg.norm(positions - pos_avg_a, axis=1) ** 2  # shape (T,)
    p_hop_list = np.zeros(shape=(T,))
    for t in range(T):
        b_interval = slice(t, T)
        a_interval = slice(0, t + 1)
        factor = np.sqrt(t * (T - t)) / T
        sq_a = np.mean(pos_valid_b_sq[a_interval], axis=0)
        sq_b = np.mean(pos_valid_a_sq[b_interval], axis=0)
        p_hop = np.sqrt(sq_a * sq_b) * factor
        p_hop_list[t] = p_hop

    return p_hop_list


def compute_non_trival_iterable(positions, threshold=0.3):
    p_hop = compute_non_trival_disp(positions)
    maxmimum, max_index = np.max(p_hop), np.argmax(p_hop)
    if maxmimum > threshold:
        positions_a = positions[: max_index + 1]
        positions_b = positions[max_index + 1 :]
        indices_a = compute_non_trival_iterable(positions_a, threshold)
        indices_b = compute_non_trival_iterable(positions_b, threshold)
        indices_b_adjusted = [idx + max_index + 1 for idx in indices_b] if indices_b else []
        return indices_a + [max_index] + indices_b_adjusted
    else:
        return []


def save_O_dict_to_h5(h5_file, strain_rate, O_dict):
    with pd.HDFStore(h5_file) as store:
        data_list = []
        for o_idx, indices in O_dict.items():
            if indices:
                for event_idx in indices:
                    data_list.append({"O_index": o_idx, "event_frame": event_idx})
        if data_list:
            df = pd.DataFrame(data_list)
            key = f"{strain_rate}"
            store.put(key, df, format="table", data_columns=True)


# 示例计算和可视化
file_paths = [
    "/home/debian/water/TIP4P/2005/2020/4096/multi/traj_1e-5_246.lammpstrj",
    # "/home/debian/water/TIP4P/2005/2020/4096/multi/traj_2.5e-5_246.lammpstrj",
    "/home/debian/water/TIP4P/2005/2020/4096/multi/traj_7.5e-5_246.lammpstrj",
    "/home/debian/water/TIP4P/2005/2020/4096/multi/traj_2.5e-4_246.lammpstrj",
]
# file_path = "/home/debian/water/TIP4P/2005/dump_H2O_246_10.lammpstrj"
for file_path in file_paths:
    u = mda.Universe(file_path, format="LAMMPSDUMP")
    O_atoms = u.select_atoms("type 1")
    coords = np.zeros(shape=(len(u.trajectory[2000:4000]), O_atoms.n_atoms, 3))
    # coords = np.zeros(shape=(len(u.trajectory), O_atoms.n_atoms, 3))
    # for ts in tqdm(u.trajectory):
    # coords[ts.frame] = O_atoms.positions.copy()
    for ts in tqdm(u.trajectory[2000:4000]):
        coords[ts.frame - 2000] = O_atoms.positions.copy()
    shear_rate = float(file_path.split("traj_")[-1].split("_246")[0]) * 1e3  # 1/ps
    time_step = 0.05  # ps
    # time_step = 0.02  # ps
    coords = apply_shear_correction(coords, shear_rate, time_step)

    O_dict = {}
    for i in tqdm(range(coords.shape[1]), desc="Computing non-trival displacements for O atoms"):
        positions = coords[:, i, :]
        O_idx = O_atoms.indices[i]
        computed_indices = compute_non_trival_iterable(positions, threshold=0.3)
        O_dict[O_idx] = computed_indices

    output_h5 = "/home/debian/water/TIP4P/2005/2020/rst/4096/non_trival_displacement_results.h5"
    save_O_dict_to_h5(output_h5, file_path.split("traj_")[-1].split("_246")[0], O_dict)
# # 做成上下两部分的图，上半部分是位移，下半部分是 non-trival displacement
# fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
# ax1.plot(displacement, label="Displacement", color="blue")
# ax1.set_ylabel("Displacement (Å)")
# ax1.legend()
# ax2.plot(non_trival_displacement, label="Non-trival Displacement", color="orange")
# ax2.axvline(x=time_interval, color="red", linestyle="--", label="Time Interval")
# ax2.axvline(
#     x=len(positions) - time_interval,
#     color="green",
#     linestyle="--",
#     label="Total Steps - Time Interval",
# )
# ax2.plot(maximum_index, maximum, "ro", label="Maximum Point")
# ax2.axhline(y=0.27, color="purple", linestyle=":", label="Threshold (0.27 Å)")
# ax2.set_ylabel("Non-trival Displacement (Å)")
# ax2.legend()
# ax2.set_xlabel("Time Step")
# plt.tight_layout()
# plt.show()
