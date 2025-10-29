import argparse
import MDAnalysis as mda
from MDAnalysis.analysis.hydrogenbonds.hbond_analysis import HydrogenBondAnalysis as HBA
from MDAnalysis.lib.distances import apply_PBC, distance_array
from MDAnalysis.lib.nsgrid import FastNS
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import os
import gc
from collections import defaultdict
import h5py


class MemoryEfficientHBAnalysis:
    """内存优化的氢键分析类"""

    def __init__(
        self, dump_file, out_dir="output", chunk_size=100, start_frame=None, end_frame=None
    ):
        self.dump_file = dump_file
        self.out_dir = out_dir
        self.chunk_size = chunk_size
        self.u = mda.Universe(dump_file, format="LAMMPSDUMP")
        self.O_atoms = self.u.select_atoms("type 1")
        self.global_O_indices = self.O_atoms.indices
        self.n_frames = len(self.u.trajectory)

        self.start_frame = start_frame if start_frame is not None else 0
        self.end_frame = end_frame if end_frame is not None else self.n_frames
        if (
            self.start_frame < 0
            or self.end_frame > self.n_frames
            or self.start_frame >= self.end_frame
        ):
            raise ValueError("Invalid start_frame or end_frame")
        # 确保输出目录存在
        os.makedirs(out_dir, exist_ok=True)

    def run_hb_analysis_chunked(self, OO_cutoff=3.5, angle_cutoff=30.0):
        """分块进行氢键分析以降低内存使用"""
        print("Running hydrogen bond analysis in chunks...")

        # 初始化HDF5文件存储结果
        hdf5_file = os.path.join(self.out_dir, "hbonds_temp.h5")

        with h5py.File(hdf5_file, "w") as h5f:
            # 创建可扩展的数据集
            hbond_dataset = h5f.create_dataset(
                "hbonds",
                (0, 6),  # frame, donor_idx, hydrogen_idx, acceptor_idx, distance, angle
                maxshape=(None, 6),
                dtype=np.float64,
                chunks=True,
                compression="gzip",
            )

            total_hbonds = 0

            # 分块处理
            for start_frame in tqdm(
                range(self.start_frame, self.end_frame + 1, self.chunk_size),
                desc="Processing chunks",
            ):
                end_frame = min(start_frame + self.chunk_size, self.end_frame + 1)

                # 对当前块进行氢键分析
                hbond_analysis = HBA(
                    universe=self.u,
                    donors_sel="type 1",
                    hydrogens_sel="type 2",
                    acceptors_sel="type 1",
                    d_a_cutoff=OO_cutoff,
                    d_h_a_angle_cutoff=180 - angle_cutoff,
                )

                hbond_analysis.run(start=start_frame, stop=end_frame)
                chunk_hbonds = hbond_analysis.results.hbonds

                if len(chunk_hbonds) > 0:
                    # 扩展数据集
                    current_size = hbond_dataset.shape[0]
                    new_size = current_size + len(chunk_hbonds)
                    hbond_dataset.resize(new_size, axis=0)

                    # 写入数据
                    hbond_dataset[current_size:new_size] = chunk_hbonds
                    total_hbonds += len(chunk_hbonds)

                # 清理内存
                del hbond_analysis, chunk_hbonds
                gc.collect()

        print(f"Total hydrogen bonds found: {total_hbonds}")
        return hdf5_file

    def process_hbond_counts_streaming(self, hdf5_file):
        """流式处理氢键计数以节省内存"""
        print("Processing hydrogen bond counts...")

        # 使用字典收集每帧的计数，避免大型DataFrame
        frame_counts = defaultdict(lambda: defaultdict(int))

        with h5py.File(hdf5_file, "r") as h5f:
            hbonds = h5f["hbonds"]

            # 分批读取数据
            batch_size = 1000
            for i in tqdm(range(0, hbonds.shape[0], batch_size), desc="Processing HB counts"):
                batch_end = min(i + batch_size, hbonds.shape[0])
                batch_data = hbonds[i:batch_end]

                for row in batch_data:
                    frame, donor_idx, _, acceptor_idx, _, _ = row
                    frame = int(frame)
                    donor_idx = int(donor_idx)
                    acceptor_idx = int(acceptor_idx)

                    frame_counts[frame][donor_idx] += 1
                    frame_counts[frame][acceptor_idx] += 1

        # 将结果写入CSV（分批写入）
        csv_file = os.path.join(self.out_dir, "hb_counts_per_idx.csv")
        with open(csv_file, "w") as f:
            f.write("frame,O_idx,hb_count\n")

            for frame in sorted(frame_counts.keys()):
                for o_idx in sorted(frame_counts[frame].keys()):
                    count = frame_counts[frame][o_idx]
                    f.write(f"{frame},{o_idx},{count}\n")

        print(f"HB counts saved to {csv_file}")
        return csv_file

    def calculate_distances_memory_efficient(self, hdf5_file):
        """内存优化的距离计算"""
        print("Calculating distances in memory-efficient manner...")

        # 预先读取氢键数据并按帧分组（使用生成器）
        def get_frame_hbonds(hdf5_file):
            """生成器：逐帧返回氢键数据"""
            with h5py.File(hdf5_file, "r") as h5f:
                hbonds = h5f["hbonds"][:]

            # 按帧分组
            frame_groups = defaultdict(list)
            for row in hbonds:
                frame = int(row[0])
                frame_groups[frame].append(row)

            return frame_groups

        frame_hbonds = get_frame_hbonds(hdf5_file)

        # 输出文件
        max_dist_file = os.path.join(self.out_dir, "max_distance_per_idx.csv")
        nhb_dist_file = os.path.join(self.out_dir, "nhb_min_distances.csv")

        with open(max_dist_file, "w") as f1, open(nhb_dist_file, "w") as f2:
            f1.write("frame,idx,max_distance\n")
            f2.write("frame,O_idx,min_distance\n")

            # 逐帧处理
            for ts in tqdm(
                self.u.trajectory[self.start_frame : self.end_frame + 1], desc="Processing frames"
            ):
                frame = ts.frame
                coords_O = self.O_atoms.positions
                box_dims = ts.dimensions

                # 应用PBC
                coords_O = apply_PBC(coords_O, box_dims)

                # 获取当前帧的氢键
                current_hbonds = frame_hbonds.get(frame, [])

                if len(current_hbonds) > 0:
                    # 计算最大氢键距离（更高效的方法）
                    max_distances = defaultdict(float)
                    hb_pairs = set()

                    for hb in current_hbonds:
                        _, donor_idx, _, acceptor_idx, distance, _ = hb
                        donor_idx = int(donor_idx)
                        acceptor_idx = int(acceptor_idx)

                        max_distances[donor_idx] = max(max_distances[donor_idx], distance)
                        max_distances[acceptor_idx] = max(max_distances[acceptor_idx], distance)
                        hb_pairs.add((donor_idx, acceptor_idx))

                    # 写入最大距离
                    for idx, max_dist in max_distances.items():
                        f1.write(f"{frame},{idx},{max_dist}\n")

                # 计算非氢键最短距离（优化算法）
                min_non_hb_distances = self._calculate_min_non_hb_distances_optimized(
                    coords_O, box_dims, hb_pairs
                )

                # 写入非氢键最短距离
                for i, min_dist in enumerate(min_non_hb_distances):
                    if min_dist < np.inf:
                        global_idx = self.global_O_indices[i]
                        f2.write(f"{frame},{global_idx},{min_dist}\n")

                # 定期清理内存
                if frame % 100 == 0:
                    gc.collect()

        print(f"Distance calculations completed")
        return max_dist_file, nhb_dist_file

    def _calculate_min_non_hb_distances_optimized(self, coords_O, box_dims, hb_pairs):
        """优化的非氢键最短距离计算"""
        n_atoms = len(coords_O)
        min_distances = np.full(n_atoms, np.inf)

        # 使用分块距离计算避免内存爆炸
        chunk_size = min(500, n_atoms)  # 根据可用内存调整

        for i in range(0, n_atoms, chunk_size):
            i_end = min(i + chunk_size, n_atoms)
            coords_i = coords_O[i:i_end]

            # 计算距离矩阵（只计算一个块与所有原子的距离）
            dist_matrix = distance_array(coords_i, coords_O, box=box_dims)

            for local_i, global_i in enumerate(range(i, i_end)):
                global_i_idx = self.global_O_indices[global_i]

                for global_j in range(n_atoms):
                    if global_i == global_j:
                        continue

                    global_j_idx = self.global_O_indices[global_j]
                    distance = dist_matrix[local_i, global_j]

                    # 检查是否为氢键对
                    if (global_i_idx, global_j_idx) not in hb_pairs and (
                        global_j_idx,
                        global_i_idx,
                    ) not in hb_pairs:
                        min_distances[global_i] = min(min_distances[global_i], distance)

        return min_distances

    def calculate_zeta_streaming(self, max_dist_file, nhb_dist_file):
        """流式计算zeta值"""
        print("Calculating zeta values...")

        # 读取数据并计算zeta（逐行处理）
        zeta_file = os.path.join(self.out_dir, "zeta.csv")

        # 首先收集所有的max_distance数据到字典
        max_distances = {}
        with open(max_dist_file, "r") as f:
            next(f)  # 跳过header
            for line in f:
                frame, idx, max_dist = line.strip().split(",")
                max_distances[(int(frame), int(idx))] = float(max_dist) * 0.1  # 转换单位

        # 处理nhb_distances并计算zeta
        with open(nhb_dist_file, "r") as f_in, open(zeta_file, "w") as f_out:
            f_out.write("frame,O_idx,distance\n")
            next(f_in)  # 跳过header

            for line in f_in:
                frame, o_idx, min_dist = line.strip().split(",")
                frame, o_idx = int(frame), int(o_idx)
                min_dist = float(min_dist) * 0.1  # 转换单位

                # 查找对应的max_distance
                max_dist = max_distances.get((frame, o_idx), 0.0)

                # 计算zeta
                zeta = min_dist - max_dist
                f_out.write(f"{frame},{o_idx},{zeta}\n")

        print(f"Zeta values saved to {zeta_file}")
        return zeta_file

    def run_complete_analysis(self, OO_cutoff=3.5, angle_cutoff=30.0):
        """运行完整的内存优化分析"""
        print("Starting memory-efficient hydrogen bond analysis...")
        print(f"Analyzing frames {self.start_frame} to {self.end_frame}...")

        # 1. 氢键分析
        hdf5_file = self.run_hb_analysis_chunked(OO_cutoff, angle_cutoff)

        # 2. 处理氢键计数
        hb_counts_file = self.process_hbond_counts_streaming(hdf5_file)

        # 3. 计算距离
        max_dist_file, nhb_dist_file = self.calculate_distances_memory_efficient(hdf5_file)

        # 4. 计算zeta值
        zeta_file = self.calculate_zeta_streaming(max_dist_file, nhb_dist_file)

        # 5. 生成分布图
        self.plot_hb_distribution(hb_counts_file)

        # 6. 清理临时文件
        if os.path.exists(hdf5_file):
            os.remove(hdf5_file)

        print("Analysis completed!")
        return {
            "hb_counts": hb_counts_file,
            "max_distances": max_dist_file,
            "nhb_distances": nhb_dist_file,
            "zeta": zeta_file,
        }

    def plot_hb_distribution(self, hb_counts_file):
        """绘制氢键分布图（内存高效版本）"""
        print("Plotting hydrogen bond distribution...")

        # 流式读取数据以计算分布
        hb_counts = []
        chunk_size = 1000

        with open(hb_counts_file, "r") as f:
            next(f)  # 跳过header
            chunk = []

            for line in f:
                _, _, count = line.strip().split(",")
                chunk.append(int(count))

                if len(chunk) >= chunk_size:
                    hb_counts.extend(chunk)
                    chunk = []

            if chunk:  # 处理最后一块
                hb_counts.extend(chunk)

        # 绘制直方图
        plt.figure(figsize=(10, 6))
        plt.hist(
            hb_counts,
            bins=range(0, max(hb_counts) + 2),
            density=True,
            alpha=0.7,
            align="left",
            rwidth=0.8,
        )
        plt.xlabel("Number of Hydrogen Bonds per Oxygen")
        plt.ylabel("Probability Density")
        plt.title("Distribution of Hydrogen Bonds per Oxygen")
        plt.grid(True, alpha=0.3)

        plot_file = os.path.join(self.out_dir, "hb_count_distribution.png")
        plt.savefig(plot_file, dpi=300, bbox_inches="tight")
        plt.show()

        print(f"Distribution plot saved to {plot_file}")


def run_memory_efficient_analysis(dump_file, out_dir):
    """主函数：运行内存优化的分析"""

    start_frame = 3500
    # 检查文件是否存在
    if not os.path.exists(dump_file):
        print(f"Trajectory file not found: {dump_file}")
        print("Please update the file path.")
        return

    # 创建分析器实例
    analyzer = MemoryEfficientHBAnalysis(
        dump_file=dump_file, out_dir=out_dir, chunk_size=50, start_frame=start_frame
    )

    # 运行完整分析
    results = analyzer.run_complete_analysis(OO_cutoff=3.5, angle_cutoff=30.0)

    print("Files generated:")
    for key, filepath in results.items():
        print(f"  {key}: {filepath}")


if __name__ == "__main__":
    # 设置内存使用限制（可选）
    import resource

    parser = argparse.ArgumentParser(description="Memory Efficient Hydrogen Bond Analysis")
    parser.add_argument(
        "--dump_file", type=str, required=False, help="Path to the LAMMPS dump file"
    )
    parser.add_argument("--out_dir", type=str, default="output", help="Output directory")

    args = parser.parse_args()
    # 限制内存使用（以字节为单位，这里设置为6GB）
    try:
        resource.setrlimit(resource.RLIMIT_AS, (6 * 1024**3, 6 * 1024**3))
    except:
        print("Warning: Could not set memory limit")

    # 运行内存优化分析
    run_memory_efficient_analysis(args.dump_file, args.out_dir)
