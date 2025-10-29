"""
计算非高斯参数（NGP）：
alpha2(t) = (3⟨r^4(t)⟩) / (5⟨r^2(t)⟩^2) - 1
"""

import numpy as np
import matplotlib.pyplot as plt
import MDAnalysis as mda
from MDAnalysis.analysis.msd import EinsteinMSD


def shear_correction(pos, y0, shear_rate, t_ps):
    corrected = pos.copy()
    corrected[:, 0] -= shear_rate * t_ps * y0 * 1e3  # shear_rate in 1/fs, t_ps in ps = fs*1e3
    return corrected


def compute_ngp(r2, r4):
    return (3 * r4) / (5 * r2**2) - 1


def main(traj_file, shear_rate=0.0, max_lag=None):
    u = mda.Universe(traj_file, format="LAMMPSDUMP")
    O_atoms = u.select_atoms("type 1")
    n_frames = len(u.trajectory)
    if max_lag is None:
        max_lag = n_frames // 10

    msd_calculator = EinsteinMSD(u, O_atoms, shear_rate=shear_rate)
