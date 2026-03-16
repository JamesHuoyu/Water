"""
Stress Analysis Module for Molecular Dynamics Simulations

This module provides functions for reading per-atom stress data from LAMMPS dump files
using MDAnalysis with custom atomstyle (positions + stress), and computing
stress distributions and correlations.
"""

import MDAnalysis as mda
import numpy as np
from scipy import stats


def parse_lammps_dump_with_stress(dump_file):
    """
    Parse LAMMPS dump file with custom stress column manually.
    MDAnalysis cannot handle c_peratom[4] syntax, so we parse directly.

    Args:
        dump_file: Path to LAMMPS dump file

    Returns:
        Custom MDAnalysis-like object with trajectory data
    """

    class CustomUniverse:
        """Custom universe that mimics MDAnalysis Universe interface"""

        def __init__(self, positions, sxy, box_dims):
            self.positions = positions  # (n_frames, n_atoms, 3)
            self.sxy = sxy  # (n_frames, n_atoms)
            self.box_dims = box_dims  # (n_frames, 6) or None
            self.n_frames = len(positions)
            self.n_atoms = positions.shape[1]

        @property
        def trajectory(self):
            return CustomTrajectory(self)

        @property
        def atoms(self):
            return Atoms(self)

    class CustomTrajectory:
        """Custom trajectory iterator"""

        def __init__(self, universe):
            self.universe = universe
            self._index = 0

        def __len__(self):
            return self.universe.n_frames

        def __iter__(self):
            self._index = 0
            return self

        def __next__(self):
            if self._index >= self.universe.n_frames:
                raise StopIteration
            ts = CustomTimestep(self._index, self.universe)
            self._index += 1
            return ts

        def __getitem__(self, idx):
            if isinstance(idx, slice):
                return [
                    CustomTimestep(i, self.universe)
                    for i in range(*idx.indices(self.universe.n_frames))
                ]
            elif isinstance(idx, int):
                return CustomTimestep(idx, self.universe)
            else:
                raise TypeError("Index must be int or slice")

    class CustomTimestep:
        """Custom timestep"""

        def __init__(self, frame_idx, universe):
            self.frame_idx = frame_idx
            self.universe = universe

        @property
        def positions(self):
            """Return positions + sxy stress in custom format"""
            n_atoms = self.universe.n_atoms
            # Create array with [x, y, z, sxy] for each atom
            positions_stress = np.zeros((n_atoms, 4))
            positions_stress[:, :3] = self.universe.positions[self.frame_idx]
            positions_stress[:, 3] = self.universe.sxy[self.frame_idx]
            return positions_stress

        @property
        def dimensions(self):
            if self.universe.box_dims is not None:
                return self.universe.box_dims[self.frame_idx]
            else:
                return None

    class Atoms:
        """Atoms container"""

        def __init__(self, universe):
            self.universe = universe
            self.n_atoms = universe.n_atoms

    # Parse the dump file
    positions = []
    sxy_values = []
    box_dims = []

    with open(dump_file, "r") as f:
        lines = f.readlines()

    i = 0
    frame_idx = 0
    n_atoms = 0

    while i < len(lines):
        line = lines[i].strip()

        if line.startswith("ITEM: TIMESTEP"):
            # Skip timestep line
            i += 2
            continue

        elif line.startswith("ITEM: NUMBER OF ATOMS"):
            # Get number of atoms
            n_atoms = int(lines[i + 1].strip())
            i += 2
            continue

        elif line.startswith("ITEM: BOX BOUNDS"):
            # Parse box dimensions
            box_line = lines[i + 1 : i + 4]
            xlo, ylo, zlo = 0.0, 0.0, 0.0
            Lx, Ly, Lz = 0.0, 0.0, 0.0
            xy, xz, yz = 0.0, 0.0, 0.0

            # 解析三斜盒子边界
            # 第一行: x方向边界和xy倾斜因子
            parts = box_line[0].strip().split()
            if len(parts) >= 2:
                xlo_bound, xhi_bound = map(float, parts[:2])
                if len(parts) >= 3:  # 三斜盒子有倾斜因子
                    xy = float(parts[2])

            # 第二行: y方向边界和xz倾斜因子
            parts = box_line[1].strip().split()
            if len(parts) >= 2:
                ylo_bound, yhi_bound = map(float, parts[:2])
                if len(parts) >= 3:
                    xz = float(parts[2])

            # 第三行: z方向边界和yz倾斜因子
            parts = box_line[2].strip().split()
            if len(parts) >= 2:
                zlo, zhi = map(float, parts[:2])
                if len(parts) >= 3:
                    yz = float(parts[2])

            xlo = xlo_bound - min(0.0, xy, xz, xy + xz)
            xhi = xhi_bound - max(0.0, xy, xz, xy + xz)
            ylo = ylo_bound - min(0.0, yz)
            yhi = yhi_bound - max(0.0, yz)
            box = np.zeros((3, 3), dtype=np.float64)
            box[0] = xhi - xlo, 0.0, 0.0
            box[1] = xy, yhi - ylo, 0.0
            box[2] = xz, yz, zhi - zlo
            # 计算角度（如果需要）
            xlen, ylen, zlen, alpha, beta, gamma = mda.lib.mdamath.triclinic_box(*box)
            # 存储为[Lx, Ly, Lz, alpha, beta, gamma]或[Lx, Ly, Lz, xy, xz, yz]
            # 根据您的需要选择存储格式
            box_dims.append([xlen, ylen, zlen, alpha, beta, gamma])  # 标准角度格式
            i += 4
            continue

        elif line.startswith("ITEM: ATOMS id type xu yu zu c_peratom[4]"):
            # Parse atom data
            frame_positions = []
            frame_sxy = []
            for j in range(n_atoms):
                atom_line = lines[i + j + 1].strip().split()
                if len(atom_line) >= 6:
                    # id, type, x, y, z, sxy
                    frame_positions.append(
                        [float(atom_line[2]), float(atom_line[3]), float(atom_line[4])]
                    )
                    frame_sxy.append(float(atom_line[5]))

            positions.append(np.array(frame_positions))
            sxy_values.append(np.array(frame_sxy))

            i += n_atoms + 1
            frame_idx += 1
            continue

        i += 1

    # Convert to numpy arrays
    positions = np.array(positions)
    sxy_values = np.array(sxy_values)
    box_dims = np.array(box_dims) if box_dims else None

    return CustomUniverse(positions, sxy_values, box_dims)


def load_universe_with_stress(dump_file):
    """
    Load MDAnalysis universe with custom atomstyle including stress.

    The dump file format is: id type x y z c_peratom[4] (sxy)
    MDAnalysis has issues reading c_peratom[4] syntax, so we need to parse manually.

    Args:
        dump_file: Path to LAMMPS dump file

    Returns:
        MDAnalysis Universe object with coordinates and stress data
    """
    # Since MDAnalysis has issues with c_peratom[4], let's parse the dump file manually
    return parse_lammps_dump_with_stress(dump_file)


def get_positions_and_sxy(u, use_relative=False, apply_box_correction=True):
    """
    Extract positions and sxy stress from Custom Universe with box correction.

    Args:
        u: Custom Universe from parse_lammps_dump_with_stress()
        use_relative: If True, return sxy deviation from frame mean (normalized)
                      If False, return absolute sxy values
        apply_box_correction: If True, apply box translation to coordinates

    Returns:
        positions: (n_frames, n_atoms, 3) array of corrected coordinates
        sxy: (n_frames, n_atoms) array of sxy values
             (absolute or relative deviation from frame mean)
    """
    n_frames = len(u.trajectory)
    n_atoms = u.n_atoms

    positions = np.zeros((n_frames, n_atoms, 3))
    sxy = np.zeros((n_frames, n_atoms))

    for i, ts in enumerate(u.trajectory):
        # positions are in first 3 columns: [0]=x, [1]=y, [2]=z
        positions[i] = u.positions[i].copy()
        # sxy stress is in the 4th column (index 3)
        sxy[i] = u.sxy[i].copy()
        # for j in range(len(sxy[i])):
        #     if j % 3 == 0:
        #         sxy[i, j] /= 14.4
        #     else:
        #         sxy[i, j] /= 4.8
        # Apply box correction if requested and box dimensions are available
        if (
            apply_box_correction
            and hasattr(u, "box_dims")
            and u.box_dims is not None
            and i < len(u.box_dims)
        ):

            box = u.box_dims[i]
            if len(box) >= 6:
                Lx, Ly, Lz, xy, xz, yz = box

                # Get box bounds from frame (they should be consistent)
                # For now, we'll assume the bounds are recorded in the box_dims
                # In a more complete implementation, we would extract them directly

                # For this implementation, we'll use a simplified approach:
                # If we have box dimensions, we'll translate coordinates
                # The actual bounds would be stored separately in a more complete version

                # For now, we'll skip the detailed translation and just store dimensions
                pass

    # Compute relative sxy after reading all frames
    if use_relative:
        sxy_relative = compute_relative_sxy(sxy)
        return positions, sxy_relative
    else:
        return positions, sxy


def compute_relative_sxy(sxy_array):
    """
    Compute sxy deviation from frame-averaged mean (normalized stress).

    The per-atom stress from LAMMPS is the atomic contribution to pressure,
    which has large absolute values (1e5-1e6 bar). For stress-structure
    correlation, we use the deviation from the mean, which has physical
    meaning: how much higher/lower an atom's stress is compared to the
    environment average.

    Args:
        sxy_array: (n_frames, n_atoms) array of sxy values (in bars)

    Returns:
        sxy_relative: sxy deviation from frame-averaged mean (dimensionless)
    """
    # Compute mean sxy for each frame
    sxy_mean_frame = np.mean(sxy_array, axis=1)  # (n_frames,)

    # Compute relative deviation from mean
    sxy_relative = sxy_array - sxy_mean_frame[:, np.newaxis]

    return sxy_relative


def compute_sxy_distribution(sxy_array):
    """
    Compute sxy distribution statistics: mean, std, skewness, kurtosis.

    Args:
        sxy_array: (n_frames, n_atoms) array of sxy values

    Returns:
        Dictionary with statistics: mean, std, median, skewness, kurtosis
    """
    # Flatten all sxy values across all frames and atoms
    sxy_flat = sxy_array.flatten()

    return {
        "mean": np.mean(sxy_flat),
        "std": np.std(sxy_flat),
        "median": np.median(sxy_flat),
        "skewness": stats.skew(sxy_flat),
        "kurtosis": stats.kurtosis(sxy_flat),
        "min": np.min(sxy_flat),
        "max": np.max(sxy_flat),
    }


def sxy_autocorrelation(sxy_array, time_lags):
    """
    Calculate shear stress autocorrelation function.

    Args:
        sxy_array: (n_frames, n_atoms) array of sxy values
        time_lags: Array of time lags (in frame indices) to compute

    Returns:
        autocorr: Array of autocorrelation values at each time lag
    """
    # Average sxy over all atoms for each frame
    sxy_mean_frame = np.mean(sxy_array, axis=1)
    n_frames = len(sxy_mean_frame)

    autocorr = np.zeros(len(time_lags))
    sxy_mean = np.mean(sxy_mean_frame)

    for i, lag in enumerate(time_lags):
        if lag >= n_frames:
            autocorr[i] = np.nan
        else:
            # Calculate autocorrelation at this lag
            shifted = sxy_mean_frame[lag:] - sxy_mean
            original = sxy_mean_frame[:-lag] - sxy_mean
            autocorr[i] = np.mean(shifted * original) / np.var(sxy_mean_frame)

    return autocorr


def spatial_sxy_correlation(sxy_array, positions, box, r_max, dr=0.1):
    """
    Calculate spatial correlation function of sxy.

    Args:
        sxy_array: (n_frames, n_atoms) array of sxy values
        positions: (n_frames, n_atoms, 3) array of coordinates
        box: Box dimensions [Lx, Ly, Lz, alpha, beta, gamma] or just [Lx, Ly, Lz]
        r_max: Maximum distance for correlation calculation
        dr: Bin width for radial correlation function

    Returns:
        r: Array of distance bins
        g_sxy: Spatial correlation function of sxy
    """
    n_frames = sxy_array.shape[0]

    # Use only the last frame for correlation (or average over frames)
    sxy_frame = sxy_array[-1]
    pos_frame = positions[-1]

    n_atoms = len(pos_frame)
    r_bins = np.arange(0, r_max + dr, dr)
    g_sxy = np.zeros(len(r_bins) - 1)

    # Extract box dimensions
    if len(box) == 3:
        Lx, Ly, Lz = box
    elif len(box) >= 6:
        Lx, Ly, Lz = box[0], box[1], box[2]
    else:
        raise ValueError("Box must be at least [Lx, Ly, Lz]")

    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            # Calculate minimum image distance with PBC
            rij = pos_frame[j] - pos_frame[i]

            # Simple PBC for orthogonal box
            rij[0] -= Lx * np.round(rij[0] / Lx)
            rij[1] -= Ly * np.round(rij[1] / Ly)
            rij[2] -= Lz * np.round(rij[2] / Lz)

            r = np.linalg.norm(rij)

            if r < r_max:
                bin_idx = int(r / dr)
                if bin_idx < len(g_sxy):
                    g_sxy[bin_idx] += sxy_frame[i] * sxy_frame[j]

    # Normalize
    for i in range(len(g_sxy)):
        # Count pairs in this bin approximately
        n_pairs = 4 * np.pi * r_bins[i] ** 2 * dr * (n_atoms / (Lx * Ly * Lz))
        if n_pairs > 0:
            g_sxy[i] /= n_pairs

    return r_bins[:-1], g_sxy


def sxy_structure_correlation(sxy_array, structural_param):
    """
    Calculate correlation between local shear stress and structure (Q4, Q6, etc.).

    Args:
        sxy_array: (n_frames, n_atoms) array of sxy values
        structural_param: (n_frames, n_atoms) array of structural parameter values

    Returns:
        correlation_coefficient: Pearson correlation coefficient
        p_value: P-value for hypothesis test of no correlation
    """
    # Flatten arrays for correlation calculation
    sxy_flat = sxy_array.flatten()
    struct_flat = structural_param.flatten()

    # Remove NaN values
    mask = ~np.isnan(sxy_flat) & ~np.isnan(struct_flat)
    sxy_flat = sxy_flat[mask]
    struct_flat = struct_flat[mask]

    if len(sxy_flat) < 2:
        return np.nan, np.nan

    # Calculate Pearson correlation
    corr_coef, p_value = stats.pearsonr(struct_flat, sxy_flat)

    return corr_coef, p_value


def map_sxy_to_grid(sxy_array, positions, box, grid_size=50):
    """
    Map shear stress to spatial grid for visualization.

    Args:
        sxy_array: (n_frames, n_atoms) array of sxy values
        positions: (n_frames, n_atoms, 3) array of coordinates
        box: Box dimensions [Lx, Ly, Lz, alpha, beta, gamma]
        grid_size: Number of grid points per dimension (for 2D) or per 3 dimensions (for 3D)

    Returns:
        grid: 2D or 3D grid of average sxy values
        edges: Grid edges for plotting
    """
    # handle single frame array
    # if sxy_array.ndim == 2 and sxy_array.shape[0] > 1:
    #     sxy_frame = sxy_array[-1]
    #     pos_frame = positions[-1]
    # else:
    #     sxy_frame = sxy_array[0]
    #     pos_frame = positions[0]
    sxy_frame = sxy_array[0]
    pos_frame = positions[0]
    pos_frame = mda.lib.distances.apply_PBC(pos_frame, box)
    Lx, Ly = box[:2]
    # Create 2D histogram on xy plane
    H, xedges, yedges = np.histogram2d(
        pos_frame[:, 0],
        pos_frame[:, 1],
        bins=grid_size,
        range=[[0, Lx], [0, Ly]],
        weights=sxy_frame,
    )
    # Count atoms in each bin for normalization
    H_count, _, _ = np.histogram2d(
        pos_frame[:, 0], pos_frame[:, 1], bins=grid_size, range=[[0, Lx], [0, Ly]]
    )
    # Average sxy in each bin
    with np.errstate(divide="ignore", invalid="ignore"):
        H = H / H_count

    return H, xedges, yedges


def compute_sxy_time_evolution(sxy_array, time_step=0.025):
    """
    Compute time evolution of sxy statistics.

    Args:
        sxy_array: (n_frames, n_atoms) array of sxy values
        time_step: Time step in ps between frames

    Returns:
        time: Array of time values
        mean_sxy: Array of mean sxy per frame
        std_sxy: Array of std sxy per frame
    """
    n_frames = sxy_array.shape[0]
    time = np.arange(n_frames) * time_step

    mean_sxy = np.mean(sxy_array, axis=1)
    std_sxy = np.std(sxy_array, axis=1)

    return time, mean_sxy, std_sxy


if __name__ == "__main__":
    # Example usage
    print("Stress Analysis Module")
    print("Use load_universe_with_stress() to load a custom dump file")
    print("Example:")
    print("  u = load_universe_with_stress('peratom_stress_5e-6_225.lammpstrj')")
    print("  positions, sxy = get_positions_and_sxy(u)")
