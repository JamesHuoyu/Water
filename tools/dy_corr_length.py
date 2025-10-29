import MDAnalysis as mda
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from MDAnalysis.analysis.distances import distance_array
from scipy.spatial.distance import pdist, squareform

# calculate g_4(r, t) for a given trajectory
# g_4(r, t) = <1/Nρ ∑_ijkl δ(r - r_k(0) + r_i(0)) w(|r_i(0) - r_j(t)|) \times w(|r_k(0) - r_l(t)|)> - <Q(t)/N>^2


def _compute_w(r1, r2, a):
    """Compute the overlap function w(r1, r2)"""
    dist = np.linalg.norm(r1 - r2, axis=-1)
    return (dist <= a).astype(float)


def _calculate_overlap_fast(positions1, positions2, box, a):
    """Fast overlap calculation using vectorized operations."""
    # Use MDAnalysis distance calculation with PBC
    distances = distance_array(positions1, positions2, box=box)

    return (distances <= a).astype(float)


def compute_g4(trajectory_file, a, r_bins, max_tau=None, stride=1):
    u = mda.Universe(trajectory_file, format="LAMMPSDUMP")
    O_atoms = u.select_atoms("type 1")
    n_water = len(O_atoms)

    total_frames = len(u.trajectory)
    if max_tau is None:
        max_tau = total_frames
    else:
        max_tau = min(max_tau, total_frames)

    frame_indices = np.arange(0, total_frames, stride)
    n_frames = len(frame_indices)
    if r_bins is None:
        r_bins = np.linspace(0, 10.0, 51)  # Default bins from 0 to 10 Å with 0.2 Å width
    n_r_bins = len(r_bins) - 1

    Q_t = np.zeros(max_tau)
    count = np.zeros(max_tau, dtype=int)

    window_size = max_tau
    position_buffer = []
    box_buffer = []

    print(f"Processing {n_frames} frames with max_tau={max_tau}")
    g4_rt = np.zeros((n_r_bins, max_tau))
    for frame_idx, global_frame in enumerate(tqdm(frame_indices, desc="Processing frames")):
        u.trajectory[global_frame]
        current_pos = O_atoms.positions.copy()
        current_box = u.trajectory.ts.dimensions.copy()
        rho = n_water / np.prod(current_box[:3])

        position_buffer.append(current_pos)
        box_buffer.append(current_box)

        if len(position_buffer) > window_size:
            position_buffer.pop(0)
            box_buffer.pop(0)

        for i, (past_pos, past_box) in enumerate(zip(position_buffer[:-1], box_buffer[:-1])):
            tau = len(position_buffer) - 1 - i
            if tau < max_tau:
                w_matrix = _calculate_overlap_fast(past_pos, current_pos, past_box, a)
                Q = np.sum(w_matrix)
                Q_t[tau] += Q
                count[tau] += 1
                # Compute g4 contributions
                past_dist = distance_array(past_pos, past_pos, box=past_box)
                for r_bin_idx in range(n_r_bins):
                    r_min = r_bins[r_bin_idx]
                    r_max = r_bins[r_bin_idx + 1]
                    r_center = 0.5 * (r_min + r_max)

                    mask_r = (past_dist >= r_min) & (past_dist < r_max)
                    if np.sum(mask_r) == 0:
                        continue

                    g4_sum = 0.0
                    # Vectorized computation of g4 contributions
                    i_indices, k_indices = np.where(mask_r)
                    for i_idx, k_idx in zip(i_indices, k_indices):
                        g4_sum += np.sum(w_matrix[i_idx, :, np.newaxis] * w_matrix[k_idx, :])

                    g4_rt[r_bin_idx, tau] += g4_sum / (n_water * rho)
    # Normalize g4_rt
    valid_mask = count > 0
    Q_t[valid_mask] /= count[valid_mask]
    for r_bin_idx in range(n_r_bins):
        g4_rt[r_bin_idx, valid_mask] /= count[valid_mask]
        g4_rt[r_bin_idx, valid_mask] -= (Q_t[valid_mask] / n_water) ** 2
    t_values = np.arange(max_tau)
    r_centers = 0.5 * (r_bins[:-1] + r_bins[1:])
    return g4_rt, Q_t, r_centers, t_values


def compute_g4_optimized(trajectory_file, a, r_bins=None, max_tau=None, stride=1, n_origins=None):
    """
    Optimized computation of g₄(r,t) with vectorized operations.

    Parameters:
    -----------
    trajectory_file : str
        Path to trajectory file
    a : float
        Overlap cutoff distance (Å)
    r_bins : np.ndarray or None
        Radial bins for g₄(r,t)
    max_tau : int or None
        Maximum time lag in frames
    stride : int
        Frame stride for efficiency
    n_origins : int or None
        Maximum number of time origins to process

    Returns:
    --------
    g4_rt : np.ndarray
        g₄(r,t) correlation function
    Q_t : np.ndarray
        Average overlap function ⟨Q(t)⟩
    r_centers : np.ndarray
        Radial bin centers
    t_values : np.ndarray
        Time values
    """
    # Load trajectory
    u = mda.Universe(trajectory_file, format="LAMMPSDUMP")
    O_atoms = u.select_atoms("type 1")
    n_water = len(O_atoms)
    total_frames = len(u.trajectory)

    print(f"Loaded trajectory: {n_water} molecules, {total_frames} frames")

    # Set defaults
    if max_tau is None:
        max_tau = min(200, total_frames // 2)

    if r_bins is None:
        r_bins = np.linspace(0, 8.0, 41)  # 0-8 Å with 0.2 Å bins

    # Calculate system density (use average over first few frames)
    densities = []
    for i in range(min(10, total_frames)):
        u.trajectory[i]
        box_volume = np.prod(u.trajectory.ts.dimensions[:3])
        densities.append(n_water / box_volume)
    rho = np.mean(densities)

    print(f"System density: {rho:.6f} molecules/Å³")

    # Set up frame processing
    available_frames = total_frames - max_tau
    if available_frames <= 0:
        raise ValueError(f"max_tau ({max_tau}) too large for trajectory ({total_frames} frames)")

    frame_indices = np.arange(0, available_frames, stride)
    if n_origins is not None:
        frame_indices = frame_indices[:n_origins]

    n_origins_actual = len(frame_indices)
    n_r_bins = len(r_bins) - 1
    r_centers = 0.5 * (r_bins[:-1] + r_bins[1:])

    print(f"Processing parameters:")
    print(f"  - Overlap cutoff: {a} Å")
    print(f"  - Max tau: {max_tau} frames")
    print(f"  - R bins: {n_r_bins} ({r_bins[0]:.1f} - {r_bins[-1]:.1f} Å)")
    print(f"  - Time origins: {n_origins_actual}")
    print(f"  - Stride: {stride}")

    # Initialize arrays
    g4_rt = np.zeros((n_r_bins, max_tau))
    Q_t = np.zeros(max_tau)
    count = np.zeros(max_tau, dtype=int)

    # Main computation loop
    for i, t0 in enumerate(tqdm(frame_indices, desc="Computing g₄(r,t)", unit="origin")):

        # Load initial frame
        u.trajectory[t0]
        r0 = O_atoms.positions.copy()
        box0 = u.trajectory.ts.dimensions.copy()

        # Precompute distance matrix at t=0 for delta function
        dist_0 = distance_array(r0, r0, box=box0)

        # Process time lags for this origin
        for tau in range(min(max_tau, total_frames - t0)):
            if t0 + tau >= total_frames:
                break

            # Load frame at t0 + tau
            u.trajectory[t0 + tau]
            r_tau = O_atoms.positions.copy()

            # Compute overlap matrix: w_ij = w(|r_i(0) - r_j(tau)|)
            w_matrix = compute_overlap_matrix(r0, r_tau, a, box0)

            # Calculate Q(t) = sum of overlaps
            Q = np.sum(w_matrix)
            Q_t[tau] += Q
            count[tau] += 1

            # Compute g₄(r,t) for each radial bin
            for r_idx in range(n_r_bins):
                r_min = r_bins[r_idx]
                r_max = r_bins[r_idx + 1]

                # Find pairs (i,k) where |r_k(0) - r_i(0)| is in [r_min, r_max]
                # This implements δ(r - |r_k(0) - r_i(0)|)
                mask_r = (dist_0 >= r_min) & (dist_0 < r_max)

                if np.sum(mask_r) == 0:
                    continue

                # Vectorized computation of quartet sum
                # Sum over all (i,j,k,l): w(r_i(0), r_j(tau)) * w(r_k(0), r_l(tau))
                # where |r_k(0) - r_i(0)| is in the current r-bin

                # Method 1: Direct but memory-intensive (for small systems)
                if n_water <= 1000:
                    # Create 4D tensor for all quartets
                    w_expanded = (
                        w_matrix[:, np.newaxis, :, np.newaxis]
                        * w_matrix[np.newaxis, :, np.newaxis, :]
                    )
                    # w_expanded[i,k,j,l] = w_ij * w_kl

                    # Apply radial mask and sum
                    quartet_sum = np.sum(w_expanded[mask_r])

                else:
                    # Method 2: Loop-based for large systems (more memory efficient)
                    quartet_sum = 0.0
                    i_indices, k_indices = np.where(mask_r)

                    for i_idx, k_idx in zip(i_indices, k_indices):
                        # For each valid (i,k) pair, sum over all (j,l)
                        quartet_sum += np.sum(w_matrix[i_idx, :, np.newaxis] * w_matrix[k_idx, :])

                # Normalize by N*ρ and number of pairs in this bin
                n_pairs_in_bin = np.sum(mask_r)
                if n_pairs_in_bin > 0:
                    g4_rt[r_idx, tau] += quartet_sum / (n_water * rho * n_pairs_in_bin)

    # Average over time origins
    valid_mask = count > 0
    Q_t_avg = np.zeros_like(Q_t)
    Q_t_avg[valid_mask] = Q_t[valid_mask] / count[valid_mask]

    # Average g4 and subtract (Q(t)/N)² term
    for r_idx in range(n_r_bins):
        g4_rt[r_idx, valid_mask] /= count[valid_mask]

    # Subtract ⟨Q(t)/N⟩² term
    Q_norm_squared = (Q_t_avg / n_water) ** 2
    for r_idx in range(n_r_bins):
        g4_rt[r_idx, :] -= Q_norm_squared

    t_values = np.arange(max_tau)
    return g4_rt, Q_t_avg, r_centers, t_values


if __name__ == "__main__":
    traj_file = "/home/debian/water/TIP4P/2005/benchmark/220/quenching/dump_H2O.lammpstrj"

    u = mda.Universe(traj_file, format="LAMMPSDUMP")
    O_atoms = u.select_atoms("type 1")
    n_water = len(O_atoms)
    n_frames = len(u.trajectory)
    print("Trajectory loaded.")

    u.trajectory[0]
    print(f"Box dimensions: {u.trajectory.ts.dimensions}")
    print(f"first 5 O atom positions:\n{O_atoms.positions[:5]}")

    if n_frames > 1000:
        print("Using a stride for memory efficiency")
        g4_rt, Q_t, r_centers, t_values = compute_g4(
            traj_file, a=1.0, r_bins=np.linspace(0, 10.0, 51), max_tau=800, stride=10
        )

    dt = 10  # ps
    t_values = t_values * dt

    results = np.column_stack((t_values, Q_t))
    np.savetxt(
        "quenching/Q_t.csv", results, delimiter=",", header="Time(ps),Q(t)", comments="", fmt="%.6e"
    )

    results_g4_at_tmax = g4_rt[:, 790]  # Example: g4 at t=7900 ps
    results = np.column_stack((r_centers, results_g4_at_tmax))
    np.savetxt(
        "quenching/g4_r_t7900ps.csv",
        results,
        delimiter=",",
        header="r(Å),g4(r,t=7900ps)",
        comments="",
        fmt="%.6e",
    )
    print("Results saved.")

    plt.figure(figsize=(10, 6))
    plt.plot(r_centers, results_g4_at_tmax, "b-", linewidth=2)
    plt.xlabel("Distance(Å)")
    plt.ylabel(r"$g_4(r, t=7900ps)$")
    plt.title(r"$g_4(r, t)$ at $t=7900ps$")
    plt.grid()
    plt.savefig("quenching/g4_r_t7900ps.png", dpi=300)
    plt.show()
