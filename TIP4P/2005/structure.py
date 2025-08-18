#!/usr/bin/env python

import numpy as np
import pandas as pd
import argparse
from pathlib import Path
from typing import List
from scipy.spatial.distance import cdist
from scipy.signal import savgol_filter
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat

import matplotlib.pyplot as plt
from tqdm import tqdm

from numba import njit, prange


@njit
def pbc_wrap(delta, L):
    """
    使用周期性边界条件处理粒子坐标
    delta: 粒子坐标差
    L: 盒子边长
    """
    delta -= np.round(delta / L) * L
    return delta


@njit(parallel=True)
def compute_rdf_all(coords, r_max, dr, V, L):
    """
    coords: (n_frames, n_particles, 3) 的粒子坐标数组
    r_max: 最大计算距离
    dr: 分箱宽度
    V: 体积
    L: 盒子边长
    """
    n_frames, n_particles, _ = coords.shape
    n_bins = int(r_max / dr)
    g = np.zeros(n_bins)

    for frame in prange(n_frames):
        # 提取当前帧的坐标
        current_coords = coords[frame]
        # 计算所有粒子对的距离
        for i in range(n_particles):
            # 只计算i<j的粒子对，避免重复计数
            for j in range(i + 1, n_particles):
                delta = current_coords[j] - current_coords[i]
                delta = pbc_wrap(delta, L)  # 应用周期性边界条件
                r = np.sqrt(delta[0] ** 2 + delta[1] ** 2 + delta[2] ** 2)

                if r < r_max:
                    bin_idx = int(r / dr)
                    if bin_idx < n_bins:
                        g[bin_idx] += 1

    # 归一化
    r_centers = np.arange(0.5 * dr, r_max, dr)
    shell_volumes = 4 * np.pi * r_centers**2 * dr
    norm_factor = n_frames * n_particles * (n_particles - 1) * shell_volumes / (2 * V)
    g = g / norm_factor
    return r_centers, g


def compute_S_q(g_r, r, dr, q, rho):
    """
    计算结构因子 S(q)
    g_r: g(r) 数组
    r: 距离数组
    dr: 分箱宽度
    q: q值
    rho: 数密度
    """
    integral = np.zeros_like(q)
    for i in range(len(r)):
        integrand = (g_r[i] - 1) * r[i] * np.sin(q * r[i]) / q
        integral += integrand * dr

    S_q = 1 + 4 * np.pi * rho * integral
    return S_q


def compute_static_structure_factor(coor1, coor2, q_vectors, L):
    """
    计算静态结构因子
    """
    q_vectors = q_vectors.T
    S_q_total = np.zeros(q_vectors.shape[-1])
    N_t, N, _ = coor1.shape
    chunk_size = 100

    for i in tqdm(range(0, N_t, chunk_size), desc="Computing S(q)"):
        chunk1 = coor1[i : i + chunk_size]
        chunk2 = coor2[i : i + chunk_size]

        # 应用周期性边界条件
        chunk1_wrapped = chunk1 - np.round(chunk1 / L) * L
        chunk2_wrapped = chunk2 - np.round(chunk2 / L) * L

        phase1 = np.sum(np.exp(1j * chunk1_wrapped @ q_vectors), axis=1)
        phase2 = np.sum(np.exp(-1j * chunk2_wrapped @ q_vectors), axis=1)
        phase = phase1 * phase2
        S_q_chunk = np.mean(phase, axis=0).real
        S_q_total += S_q_chunk * chunk_size

    S_q = S_q_total / (N_t * N)
    return S_q


def compute_selfstatic_structure_factor(coordinates, q_vectors, L):
    """
    计算自静态结构因子
    """
    q_vectors = q_vectors.T
    S_q_total = np.zeros(q_vectors.shape[-1])
    N_t, N, _ = coordinates.shape
    chunk_size = 50

    for i in tqdm(range(0, coordinates.shape[0], chunk_size), desc="Computing S(q)"):
        chunk = coordinates[i : i + chunk_size]
        # 应用周期性边界条件
        chunk_wrapped = chunk - np.round(chunk / L) * L
        phase = np.exp(1j * chunk_wrapped @ q_vectors)
        phase_sum = np.sum(phase, axis=1)
        S_q_chunk = np.mean(np.abs(phase_sum) ** 2, axis=0) / N
        S_q_total += S_q_chunk * chunk_size

    S_q = S_q_total / N_t
    return S_q


def compute_ISF(coordinates, q_vectors, L):
    """
    计算中间散射函数（ISF）
    coordinates: 粒子坐标数组
    q_vectors: q矢量数组
    L: 盒子边长
    """
    N_t, N, _ = coordinates.shape
    ISF_list = []
    q_norms = np.linalg.norm(q_vectors, axis=1)

    # 只计算一个q值以简化
    q = q_vectors[0]
    q_norm = q_norms[0]

    for t in tqdm(range(1, min(100, N_t)), desc="Computing ISF"):
        ISF_total = 0
        N_t_window = min(1000, N_t - t)

        for t0 in range(N_t_window):
            curr_coords = coordinates[t0]
            next_coords = coordinates[t0 + t]
            diff = next_coords - curr_coords
            # 应用周期性边界条件
            diff = pbc_wrap(diff, L)
            phase = np.cos(np.sum(diff * q, axis=1))
            ISF = np.mean(phase)
            ISF_total += ISF

        ISF_total /= N_t_window
        ISF_list.append(ISF_total)

    return np.array(ISF_list), q_norm


def main(file_patterns: List[str], output_path: Path):
    input_files = []
    for pattern in file_patterns:
        input_files.extend(sorted(Path().glob(pattern)))
    print(input_files)

    coordinates_list = []
    for file in input_files:
        coordinates = np.load(file, mmap_mode="r")
        coordinates_list.append(coordinates)

    # 参数设置
    r_max = 12.4377  # 最大计算距离
    dr = 0.1  # 分箱宽度
    L = 24.87541  # 盒子边长
    N = 512  # 粒子数
    V = L**3  # 盒子体积
    rho = N / V  # 数密度

    # 创建输出目录
    output_path.mkdir(parents=True, exist_ok=True)

    # 计算 g(r)
    print("计算径向分布函数 g(r)...")
    results = []
    for coords in tqdm(coordinates_list):
        r, g_r = compute_rdf_all(coords, r_max, dr, V, L)
        results.append((r, g_r))

    # 保存结果
    for idx, (r, g_r) in enumerate(results):
        np.save(output_path / f"g_r_{idx}.npy", np.stack([r, g_r]))
        plt.plot(r, g_r, label=f"File {idx+1}")

    plt.xlabel("r")
    plt.ylabel("g(r)")
    plt.title("Radial Distribution Function")
    plt.legend()
    plt.savefig(output_path / "g_r_comparison.png")
    plt.close()

    # 计算结构因子 S(q)
    print("计算结构因子 S(q)...")
    q_min, q_max = 0.1, 25.0
    n_q = 100  # q的数量
    q_values = np.linspace(q_min, q_max, n_q)

    S_q_results = []
    for r, g_r in results:
        S_q = np.zeros_like(q_values)
        for i, q in enumerate(q_values):
            S_q[i] = compute_S_q(g_r, r, dr, q, rho)
        S_q_results.append(S_q)
        np.save(output_path / f"S_q_{idx}.npy", S_q)
        plt.plot(q_values, S_q, label=f"File {idx+1}")

    plt.xlabel("q")
    plt.ylabel("S(q)")
    plt.title("Structure Factor")
    plt.legend()
    plt.savefig(output_path / "S_q_comparison.png")
    plt.close()

    # 计算中间散射函数 ISF
    print("计算中间散射函数 ISF...")
    q_vectors = np.array([[0, 0, 7.16]])  # 使用单个q值简化计算

    ISF_results = []
    for idx, coords in enumerate(coordinates_list):
        ISF, q_norm = compute_ISF(coords, q_vectors, L)
        np.save(output_path / f"ISF_{idx}.npy", ISF)
        time_points = np.arange(1, len(ISF) + 1)
        plt.plot(time_points, ISF, label=f"q = {q_norm:.2f} Å⁻¹ (File {idx+1})")

    plt.xlabel("Time")
    plt.ylabel("ISF")
    plt.title("Intermediate Scattering Function")
    plt.legend()
    plt.savefig(output_path / "ISF_comparison.png")
    plt.close()

    print("分析完成！结果保存在:", output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("patterns", nargs="+", help="Input file patterns (e.g., '*.npy')")
    parser.add_argument("--output_path", type=Path, default=Path("output"), help="Output path")
    args = parser.parse_args()
    main(args.patterns, args.output_path)
