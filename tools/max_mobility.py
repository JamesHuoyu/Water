import numpy as np
import matplotlib.pyplot as plt
import MDAnalysis as mda
import pandas as pd
from tqdm import tqdm
from numba import jit, prange


class MaxMobilityCalculator:
    def __init__(
        self,
        universe: mda.Universe,
        shear_rate: float = 0.0,
        time_step: float = 1.0,
        start_index: int = 0,
    ):
        self.universe = universe
        self.n_frames = len(universe.trajectory)
        self.n_particles = len(universe.atoms)
        self.O_atoms = self.universe.select_atoms("type 1")
        # 预加载轨迹数据到内存，只针对O原子
        self.frames = self.n_frames - start_index
        self.coords = np.zeros((self.frames, len(self.O_atoms), 3))
        for ts in tqdm(self.universe.trajectory[start_index:], desc="Loading trajectory data"):
            self.coords[ts.frame - start_index] = self.O_atoms.positions.copy()
        if shear_rate != 0.0:
            self.shear_correction(shear_rate, time_step)

    def shear_correction(self, shear_rate, time_step):
        for frame in tqdm(range(self.frames), desc="Applying shear correction"):
            if frame == 0:
                continue
            y_positions = self.coords[frame - 1, :, 1]
            # 修正x坐标以消除剪切流影响
            self.coords[frame:, :, 0] -= shear_rate * time_step * y_positions

    @staticmethod
    @jit(nopython=True, parallel=True)
    def compute_displacements_numba(t0, delta_t, positions, n_frames, n_atoms):
        if t0 + delta_t >= n_frames:
            return None, None
        disp = np.zeros((n_atoms, 3))
        for i in prange(n_atoms):
            start_pos = positions[t0, i]
            end_pos = positions[t0 + delta_t, i]
            disp[i] = end_pos - start_pos
        disp_root_squared = np.sqrt(np.sum(disp**2, axis=1))  # shape (n_atoms,)
        return disp, disp_root_squared

    def calculate_max_mobility_for_one_origin(self, t0: int, delta_t: int):
        return self.compute_displacements_numba(
            t0, delta_t, self.coords, self.frames, len(self.O_atoms)
        )

    def time_origin_average(self, delta_t: int):
        n_origins = self.frames - delta_t
        disp_accum = np.zeros((len(self.O_atoms), 3))
        dist = np.zeros(len(self.O_atoms))
        for t0 in tqdm(range(n_origins), desc="Calculating time origin average"):
            disp, disp_root_squared = self.calculate_max_mobility_for_one_origin(t0, delta_t)
            disp_accum += disp
            dist += disp_root_squared
        return disp_accum / n_origins, dist / n_origins


# 示例使用
if __name__ == "__main__":
    pathfiles = [
        # "/home/debian/water/TIP4P/2005/2020/4096/multi/traj_2.5e-4_246.lammpstrj",
        # "/home/debian/water/TIP4P/2005/2020/4096/multi/traj_7.5e-5_246.lammpstrj",
        "/home/debian/water/TIP4P/2005/2020/4096/multi/traj_2.5e-5_246.lammpstrj",
        # "/home/debian/water/TIP4P/2005/2020/4096/multi/traj_1e-5_246.lammpstrj",
    ]
    # pathfiles = ["/home/debian/water/TIP4P/2005/2020/4096/traj_2.5e-4_246_everystep.lammpstrj"]
    # pathfiles = ["/home/debian/water/TIP4P/2005/dump_H2O_246_10.lammpstrj"]
    output_h5 = "/home/debian/water/TIP4P/2005/2020/rst/4096/new_mobility_results.h5"
    # output_h5 = "test_msd_results.h5"
    store = pd.HDFStore(output_h5)

    start_index = 2000  # 跳过前2000帧以避免初始非平衡影响
    t_x = 3.0  # ps
    for pathfile in pathfiles:
        time_step = 0.05  # ps
        target_frame = int(t_x / time_step)
        u = mda.Universe(pathfile, format="LAMMPSDUMP")
        mobility_calculator = MaxMobilityCalculator(
            u, shear_rate=2.5e-2, time_step=time_step
        )  # shear_rate in 1/ps(2.5e-5 1/fs)
        # msd_calculator = MSDCalculator(u, start_index=start_index)  # 无剪切流
        disp, dist = mobility_calculator.time_origin_average(target_frame)
        times = np.arange(len(dist)) * time_step
        idx = np.arange(len(dist))

        df = pd.DataFrame({"O_idx": idx, "avg_dist": dist})
        filename = pathfile.split("traj_")[-1].split("_246")[0]
        # filename = "1e-5-x"
        store.put(filename, df, format="table")
        print(f"Saved mobility results to key: {filename}")
    store.close()
    print(f"All mobility results saved to: {output_h5}")
