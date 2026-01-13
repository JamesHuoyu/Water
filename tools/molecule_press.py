import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import MDAnalysis as mda
from tqdm import tqdm
from scipy.spatial import Voronoi, ConvexHull


# Calculate voronoi cell volumes
class MSDCalculator:
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

    def calculate_voronoi_volumes(self, positions):
        vor = Voronoi(positions)
        volumes = []
        for region in vor.regions:
            if not -1 in region and len(region) > 0:
                vertices = vor.vertices[region]
                hull = ConvexHull(vertices)
                volumes.append(hull.volume)
            else:
                volumes.append(np.nan)
        return np.array(volumes)
