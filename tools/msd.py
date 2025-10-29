import MDAnalysis as mda
import pandas as pd
import numpy as np
import argparse
from numba import jit, prange
from tqdm import tqdm


class MSDCalculator:
    def __init__(self, universe: mda.Universe, shear_rate: float = 0.0, time_step: float = 1.0):
        self.universe = universe
        self.n_frames = len(universe.trajectory)
        self.n_particles = len(universe.atoms)
        self.O_atoms = self.universe.select_atoms("type 1")
        # 预加载轨迹数据到内存，只针对O原子
        self.coords = np.zeros((self.n_frames, len(self.O_atoms), 3))
        for ts in tqdm(self.universe.trajectory, desc="Loading trajectory data"):
            self.coords[ts.frame] = self.O_atoms.positions.copy()
        if shear_rate != 0.0:
            self.shear_correction(shear_rate, time_step)

    def shear_correction(self, shear_rate, time_step):
        for frame in tqdm(range(self.n_frames), desc="Applying shear correction"):
            if frame == 0:
                continue
            y_positions = self.coords[frame - 1, :, 1]
            # 修正x坐标以消除剪切流影响
            self.coords[frame:, :, 0] -= shear_rate * time_step * y_positions

    @staticmethod
    @jit(nopython=True, parallel=True, fastmath=True)
    def compute_msd_numba(zs, t0, max_tau, n_particles):
        n_tau = min(zs.shape[0] - t0, max_tau)
        msd_t0 = np.zeros(n_tau)
        ref_pos = zs[t0]

        for tau in prange(n_tau):
            current_pos = zs[t0 + tau]
            dr = current_pos - ref_pos
            sq_disp = 0.0
            for i in range(n_particles):
                disp = dr[i] ** 2
                sq_disp += disp
            msd_t0[tau] = sq_disp / n_particles
        return msd_t0

    def compute_msd_for_origin(self, t0: int) -> np.ndarray:
        # 计算x方向的MSD
        return self.compute_msd_numba(
            self.coords[:, :, 0], t0, self.n_frames - t0, len(self.O_atoms)
        )

    def time_origin_average(self, max_tau: int = None) -> np.ndarray:
        if max_tau is None:
            max_tau = self.n_frames
        msd_accum = np.zeros(max_tau)
        count = np.zeros(max_tau, dtype=int)

        for t0 in tqdm(range(self.n_frames), desc="Computing MSD time-origin average"):
            msd_t0 = self.compute_msd_for_origin(t0)
            valid_tau = min(len(msd_t0), max_tau)

            msd_accum[:valid_tau] += msd_t0[:valid_tau]
            count[:valid_tau] += 1

        valid_mask = count > 0
        msd_avg = np.zeros(max_tau)
        msd_avg[valid_mask] = msd_accum[valid_mask] / count[valid_mask]
        return msd_avg


if __name__ == "__main__":
    pathfiles = [
        # "/home/debian/water/TIP4P/2005/2020/4096/multi/traj_2.5e-4_246.lammpstrj"
        "/home/debian/water/TIP4P/2005/2020/4096/traj_2.5e-5_246.lammpstrj",
        # "/home/debian/water/TIP4P/2005/2020/4096/multi/traj_2.5e-6_246.lammpstrj",
        # "/home/debian/water/TIP4P/2005/2020/4096/multi/traj_5e-6_246.lammpstrj",
        # "/home/debian/water/TIP4P/2005/2020/4096/multi/traj_5e-7_246.lammpstrj",
    ]
    # pathfiles = ["/home/debian/water/TIP4P/2005/2020/4096/traj_2.5e-5_246_everystep.lammpstrj"]
    # pathfiles = ["/home/debian/water/TIP4P/2005/dump_H2O_246_10.lammpstrj"]
    output_h5 = "/home/debian/water/TIP4P/2005/2020/rst/4096/all_msd_results.h5"
    # output_h5 = "test_msd_results.h5"
    store = pd.HDFStore(output_h5)

    for pathfile in pathfiles:
        time_step = 0.5  # ps
        u = mda.Universe(pathfile, format="LAMMPSDUMP")
        msd_calculator = MSDCalculator(
            u, shear_rate=2.5e-2, time_step=time_step
        )  # shear_rate in 1/ps(2.5e-5 1/fs)
        # msd_calculator = MSDCalculator(u)  # 无剪切流
        msd = msd_calculator.time_origin_average()
        times = np.arange(len(msd)) * time_step

        df = pd.DataFrame({"time_ps": times, "MSD_A2": msd})
        # filename = pathfile.split("traj_")[-1].split("_246")[0]
        filename = "2.5e-5-x"
        store.put(filename, df, format="table")
        print(f"Saved MSD results to key: {filename}")
    store.close()
    print(f"All MSD results saved to: {output_h5}")
