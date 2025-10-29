import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis import rdf
from MDAnalysis.lib.nsgrid import FastNS
from MDAnalysis.lib.distances import distance_array, apply_PBC
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

# 2.5e-5/2.5e-4 第二壳层5.6，高点4.4


class Classifier:
    def __init__(self, universe: mda.Universe, hb_count: pd.DataFrame = None):
        self.universe = universe
        self.n_atoms = len(universe.atoms)
        self.O_atoms = self.universe.select_atoms("type 1")
        self.global_O_indices = self.O_atoms.indices
        self.hb_count = hb_count
        # 载入与hb_count对应的帧数据
        self.start_frame = self.hb_count["frame"].min() if hb_count is not None else 0
        self.end_frame = (
            self.hb_count["frame"].max() if hb_count is not None else len(universe.trajectory) - 1
        )
        print(f"Classifying frames from {self.start_frame} to {self.end_frame}")
        self.positions = np.zeros((self.end_frame - self.start_frame + 1, len(self.O_atoms), 3))
        for ts in tqdm(
            self.universe.trajectory[self.start_frame : self.end_frame + 1],
            desc="Loading positions",
        ):
            self.positions[ts.frame - self.start_frame] = self.O_atoms.positions.copy()

    def get_neighbors_within_radius(self, frame: int, radius: float = 5.6):
        """获取指定帧中每个水分子在给定半径内的邻居索引列表"""
        frame = frame - self.start_frame  # 调整为positions数组的索引
        if frame < 0 or frame >= len(self.positions):
            raise ValueError("帧索引超出范围")
        coords = self.positions[frame].astype(np.float32)
        box = self.universe.dimensions.astype(np.float32)  # 获取盒子尺寸

        searcher = FastNS(radius, coords, box, pbc=True)
        results = searcher.search(coords)

        neighbors_list = {i: [] for i in self.global_O_indices}
        for i, j in zip(results.get_pairs()[:, 0], results.get_pairs()[:, 1]):
            if i != j:
                i = self.global_O_indices[i]
                j = self.global_O_indices[j]
                neighbors_list[i].append(j)
        # print(neighbors_list)
        return neighbors_list

    def classify_by_hb(self, frame: int):
        """根据自身氢键数量以及周围第二水合层内的所有水分子的氢键数量对水分子进行分类"""
        if self.hb_count is None:
            raise ValueError("氢键计数数据未提供")

        # 获取当前帧的氢键计数
        current_hb = self.hb_count[self.hb_count["frame"] == frame]
        if current_hb.empty:
            raise ValueError(f"帧 {frame} 的氢键数据不可用")
        neighbors = self.get_neighbors_within_radius(frame, radius=5.6)
        for i, row in current_hb.iterrows():
            hb_self = row["hb_count"]
            neighbor_indices = neighbors[row["O_idx"]]
            hb_neighbors = current_hb[current_hb["O_idx"].isin(neighbor_indices)]["hb_count"]
            if np.all(hb_neighbors == 4) and hb_self == 4:
                self.hb_count.at[i, "class"] = "Tetrahedral"
                print(f"O_idx {row['O_idx']} classified as Tetrahedral")
            else:
                self.hb_count.at[i, "class"] = "Other"
        return self.hb_count


# 示例使用
u = mda.Universe(
    "/home/debian/water/TIP4P/2005/2020/4096/traj_2.5e-5_246.lammpstrj", format="LAMMPSDUMP"
)
# 假设已经有氢键计数数据
hb_data = pd.read_csv("/home/debian/water/TIP4P/2005/2020/rst/4096/2.5e-5/hb_counts_per_idx.csv")

classifier = Classifier(u, hb_count=hb_data)
classified_data = classifier.classify_by_hb(frame=3700)

# print(classified_data[classified_data["frame"] == 3700])
