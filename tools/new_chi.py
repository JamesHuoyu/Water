#!/usr/bin/env python3
import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis.distances import distance_array
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import argparse


def compute_overlap(r0, r1, box, a):
    """计算每个原子是否在 cutoff a 内保持重叠"""
    dr = distance_array(r0, r1, box=box)
    diag_dist = np.diag(dr)
    return np.mean(diag_dist <= a)


def compute_chi4(
    trajectory,
    a=1.0,
    stride=1,
    max_tau=None,
    dt_ps=5.0,
    start_frame=0,
    end_frame=None,
    out_prefix="chi4",
):
    """
    计算四点易感性 χ₄(t)

    Parameters
    ----------
    trajectory : str
        LAMMPS dump 文件路径 (type 1 为 O 原子)
    a : float
        Overlap cutoff (Å)
    stride : int
        每隔多少帧计算一次（提高效率）
    max_tau : int or None
        最大延迟帧数（若为 None 则用全部）
    dt_ps : float
        每帧的物理时间间隔（单位 ps）
    start_frame : int
        起始帧（默认 0）
    end_frame : int or None
        结束帧（若为 None 则处理到最后）
    out_prefix : str
        输出文件前缀
    """

    # --- 读取轨迹 ---
    u = mda.Universe(trajectory, format="LAMMPSDUMP")
    O_atoms = u.select_atoms("type 1")
    nO = len(O_atoms)
    total_frames = len(u.trajectory)
    print(f"读取轨迹: {trajectory}")
    print(f"共有 {total_frames} 帧，包含 {nO} 个 O 原子")
    if end_frame is None or end_frame > total_frames:
        end_frame = total_frames
    total_frames = end_frame - start_frame
    if max_tau is None:
        max_tau = total_frames
    else:
        max_tau = min(max_tau, total_frames)

    print(f"Overlap cutoff = {a:.3f} Å, stride = {stride}, max_tau = {max_tau}")

    # --- 初始化数据 ---
    Q_t = np.zeros(max_tau + 1)
    Q_t2 = np.zeros(max_tau + 1)
    count = np.zeros(max_tau + 1, dtype=int)

    # --- 预加载所有帧（若轨迹较小） ---
    print("加载所有帧中...")
    positions = []
    boxes = []
    for ts in tqdm(u.trajectory[start_frame:end_frame], desc="Loading"):
        positions.append(O_atoms.positions.copy())
        boxes.append(ts.dimensions.copy())

    positions = np.array(positions)
    boxes = np.array(boxes)

    # --- 主循环 ---
    print("计算 χ₄(t)...")
    for i0 in tqdm(range(0, total_frames, stride), desc="Frames"):
        r0 = positions[i0]
        box0 = boxes[i0]
        for i1 in range(i0 + 1, min(i0 + 1 + max_tau, total_frames), stride):
            dt = i1 - i0
            Q = compute_overlap(r0, positions[i1], boxes[i1], a)
            Q_t[dt] += Q
            Q_t2[dt] += Q * Q
            count[dt] += 1

    # --- 归一化 ---
    valid = count > 0
    Q_t[valid] /= count[valid]
    Q_t2[valid] /= count[valid]

    chi4 = np.zeros_like(Q_t)
    chi4[valid] = (Q_t2[valid] - Q_t[valid] ** 2) * nO
    t_ps = np.arange(max_tau + 1) * dt_ps

    # --- 保存结果 ---
    out_csv = f"{out_prefix}_chi4.csv"
    np.savetxt(
        out_csv,
        np.column_stack((t_ps, chi4)),
        header="time_ps,chi4",
        delimiter=",",
        fmt="%.6e",
        comments="",
    )
    print(f"结果已保存至 {out_csv}")

    # --- 绘图 ---
    plt.figure(figsize=(7, 5))
    plt.plot(t_ps[valid], chi4[valid], "o-", lw=1.2, ms=3)
    plt.xscale("log")
    plt.xlabel("t (ps)")
    plt.ylabel(r"$\chi_4(t)$")
    plt.title("Four-point Susceptibility")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_chi4.png", dpi=300)
    print(f"图像已保存至 {out_prefix}_chi4.png")

    # --- 返回数据 ---
    return t_ps, chi4


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute four-point susceptibility χ₄(t).")
    parser.add_argument("--traj", required=True, help="LAMMPS dump trajectory file")
    parser.add_argument("--a", type=float, default=1.0, help="Overlap cutoff distance (Å)")
    parser.add_argument("--stride", type=int, default=5, help="Frame stride")
    parser.add_argument("--max_tau", type=int, default=None, help="Max τ (frames)")
    parser.add_argument("--dt", type=float, default=5.0, help="Frame interval (ps)")
    parser.add_argument("--start_frame", type=int, default=0, help="Start frame")
    parser.add_argument("--end_frame", type=int, default=None, help="End frame")
    parser.add_argument("--out", default="chi4", help="Output file prefix")

    args = parser.parse_args()

    compute_chi4(
        trajectory=args.traj,
        a=args.a,
        stride=args.stride,
        max_tau=args.max_tau,
        dt_ps=args.dt,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        out_prefix=args.out,
    )
