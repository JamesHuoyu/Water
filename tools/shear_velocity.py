import MDAnalysis as mda
import pandas as pd
from tqdm import tqdm
from MDAnalysis import Universe
from numba import jit, prange
import numpy as np


# 2.5e-5,7.5e-5 是正常的，1e-5有问题（5埃的地方为原点）
class ShearVelocityAnalyzer:
    def __init__(self, universe: Universe, shear_rate: float = 0.0, time_step: float = 1.0):
        self.universe = universe
        self.n_frames = len(universe.trajectory)
        self.n_particles = len(universe.atoms)
        self.O_atoms = self.universe.select_atoms("type 1")
        # 预加载轨迹数据到内存，只针对O原子
        self.coords = np.zeros((self.n_frames, len(self.O_atoms), 3))
        for ts in tqdm(self.universe.trajectory, desc="Loading trajectory data"):
            self.coords[ts.frame] = self.O_atoms.positions.copy()
        self.velocities = np.zeros((self.n_frames, len(self.O_atoms), 3))
        for ts in tqdm(self.universe.trajectory, desc="Loading velocity data"):
            self.velocities[ts.frame] = self.O_atoms.velocities.copy()
        # if shear_rate != 0.0:
        #     self.shear_correction(shear_rate, time_step)

    def shear_correction(self, shear_rate, time_step):
        for frame in tqdm(range(self.n_frames), desc="Applying shear correction"):
            if frame == 0:
                continue
            y_positions = self.coords[frame - 1, :, 1]
            self.coords[frame:, :, 0] -= shear_rate * time_step * y_positions

    def plot_vx_profile(self):
        "计算并绘制vx vs y的速度剖面，返回y位置和对应的平均vx速度"
        # 选择特定帧（例如第3600帧）的数据进行分析
        frame = 3600
        self.universe.trajectory[frame]
        _, Ly, _, _, _, theta = self.universe.dimensions
        height = Ly * np.sin(np.radians(theta))
        print(f"Box height in y-direction: {height} Å")
        vx_all = self.velocities[frame:, :, 0].flatten()
        y_all = self.coords[frame:, :, 1].flatten()
        y_bins = np.linspace(0, height, 100)
        vx_profile = np.zeros(len(y_bins) - 1)
        for i in range(len(y_bins) - 1):
            mask = (y_all >= y_bins[i]) & (y_all < y_bins[i + 1])
            if np.sum(mask) > 0:
                vx_profile[i] = np.mean(vx_all[mask])
            else:
                vx_profile[i] = 0.0
        y_centers = 0.5 * (y_bins[:-1] + y_bins[1:])
        return y_centers, vx_profile


# 示例使用
u = mda.Universe(
    "/home/debian/water/TIP4P/2005/2020/4096/traj_2.5e-5_246.lammpstrj", format="LAMMPSDUMP"
)
analyzer = ShearVelocityAnalyzer(u)  # shear_rate in 1/ps
y_positions, vx_profile = analyzer.plot_vx_profile()
import matplotlib.pyplot as plt

plt.figure(figsize=(7, 5))
plt.plot(y_positions, vx_profile, "-o")
plt.xlabel("Position in y (Å)")
plt.ylabel("Velocity in x (Å/fs)")
plt.title("Shear Velocity Profile")
plt.grid(True)
plt.show()
