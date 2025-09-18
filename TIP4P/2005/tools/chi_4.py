import numpy as np
import matplotlib.pyplot as plt
import MDAnalysis as mda
from MDAnalysis.analysis.distances import distance_array
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import os


def calculate_chi4_optimized(trajectory_file, a=0.1, stride=1, max_frames=None, n_processes=None):
    """
    Optimized calculation of four-point susceptibility chi4.

    Parameters:
    -----------
    trajectory_file : str
        Path to trajectory file
    a : float
        Cutoff distance for overlap function
    stride : int
        Skip every nth frame to reduce computation
    max_frames : int or None
        Maximum number of frames to process
    n_processes : int or None
        Number of processes for parallelization (default: cpu_count())
    """
    if n_processes is None:
        n_processes = min(cpu_count(), 4)  # Don't use too many processes

    u = mda.Universe(trajectory_file, format="LAMMPSDUMP")
    O_atoms = u.select_atoms("type 1")
    n_water = len(O_atoms)

    # Determine frames to process
    total_frames = len(u.trajectory)
    if max_frames:
        total_frames = min(total_frames, max_frames)

    frame_indices = np.arange(0, total_frames, stride)
    n_frames = len(frame_indices)

    print(f"Processing {n_frames} frames (every {stride} frames)")
    print(f"Using {n_processes} processes")

    # Initialize arrays
    Q_t = np.zeros(n_frames)
    Q_t2 = np.zeros(n_frames)
    count = np.zeros(n_frames, dtype=int)

    # Process in chunks to manage memory
    chunk_size = min(100, n_frames)  # Process 100 frames at a time

    for chunk_start in tqdm(range(0, n_frames, chunk_size), desc="Processing chunks"):
        chunk_end = min(chunk_start + chunk_size, n_frames)
        chunk_indices = frame_indices[chunk_start:chunk_end]

        # Load positions for this chunk
        positions = []
        boxes = []

        for idx in chunk_indices:
            u.trajectory[idx]
            positions.append(O_atoms.positions.copy())
            boxes.append(u.trajectory.ts.dimensions.copy())

        positions = np.array(positions)
        boxes = np.array(boxes)

        # Calculate chi4 for this chunk
        chunk_Q_t, chunk_Q_t2, chunk_count = _process_chunk(
            positions, boxes, a, n_water, chunk_start, n_frames
        )

        Q_t += chunk_Q_t
        Q_t2 += chunk_Q_t2
        count += chunk_count

    # Average and normalize
    valid_mask = count > 0
    Q_t[valid_mask] /= count[valid_mask]
    Q_t2[valid_mask] /= count[valid_mask]

    # Calculate chi4
    chi4_values = np.zeros_like(Q_t)
    chi4_values[valid_mask] = (Q_t2[valid_mask] - Q_t[valid_mask] ** 2) / (n_water**2)

    t_values = np.arange(n_frames) * stride
    return chi4_values, t_values


def _process_chunk(positions, boxes, a, n_water, chunk_start, total_frames):
    """Process a chunk of frames for chi4 calculation."""
    chunk_size = len(positions)
    Q_t = np.zeros(total_frames)
    Q_t2 = np.zeros(total_frames)
    count = np.zeros(total_frames, dtype=int)

    for i in range(chunk_size):
        t0_global = chunk_start + i
        r0 = positions[i]
        box0 = boxes[i]

        # Only calculate for future times to avoid double counting
        for j in range(i, chunk_size):
            t_global = chunk_start + j
            dt = t_global - t0_global

            if dt < total_frames:
                rt = positions[j]

                # Calculate overlap more efficiently
                Q = _calculate_overlap_fast(r0, rt, box0, a)

                Q_t[dt] += Q
                Q_t2[dt] += Q**2
                count[dt] += 1

    return Q_t, Q_t2, count


def _calculate_overlap_fast(r0, rt, box, a):
    """Fast overlap calculation using vectorized operations."""
    # Use MDAnalysis distance calculation with PBC
    distances = distance_array(r0, rt, box=box)

    # Count overlaps (distances <= a)
    overlaps = np.sum(distances <= a)
    return overlaps


def calculate_chi4_memory_efficient(trajectory_file, a=0.1, stride=1, max_tau=None):
    """
    Memory-efficient version that processes trajectory on-the-fly.

    Parameters:
    -----------
    max_tau : int or None
        Maximum time difference to calculate (reduces computation)
    """
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

    Q_t = np.zeros(max_tau)
    Q_t2 = np.zeros(max_tau)
    count = np.zeros(max_tau, dtype=int)

    # Store positions for sliding window
    window_size = max_tau
    position_buffer = []
    box_buffer = []

    print(f"Processing {n_frames} frames with max_tau={max_tau}")

    for frame_idx, global_frame in enumerate(tqdm(frame_indices, desc="Processing frames")):
        u.trajectory[global_frame]
        current_pos = O_atoms.positions.copy()
        current_box = u.trajectory.ts.dimensions.copy()

        # Add to buffer
        position_buffer.append(current_pos)
        box_buffer.append(current_box)

        # Maintain buffer size
        if len(position_buffer) > window_size:
            position_buffer.pop(0)
            box_buffer.pop(0)

        # Calculate overlaps with previous frames in buffer
        for i, (past_pos, past_box) in enumerate(zip(position_buffer[:-1], box_buffer[:-1])):
            tau = len(position_buffer) - 1 - i
            if tau < max_tau:
                Q = _calculate_overlap_fast(past_pos, current_pos, past_box, a)
                Q_t[tau] += Q
                Q_t2[tau] += Q**2
                count[tau] += 1

    # Average and normalize
    valid_mask = count > 0
    Q_t[valid_mask] /= count[valid_mask]
    Q_t2[valid_mask] /= count[valid_mask]

    chi4_values = np.zeros_like(Q_t)
    chi4_values[valid_mask] = (Q_t2[valid_mask] - Q_t[valid_mask] ** 2) / (n_water**2)

    t_values = np.arange(max_tau) * stride
    return chi4_values, t_values


if __name__ == "__main__":
    trajectory_file = "/home/debian/water/TIP4P/2005/benchmark/220/quenching/dump_H2O.lammpstrj"

    # Check if file exists
    if not os.path.exists(trajectory_file):
        print(f"Trajectory file not found: {trajectory_file}")
        print("Please update the file path or create a sample trajectory for testing.")
        exit(1)

    # Load and inspect trajectory
    u = mda.Universe(trajectory_file, format="LAMMPSDUMP")
    O_atoms = u.select_atoms("type 1")
    n_frames = len(u.trajectory)
    n_water = len(O_atoms)

    print("Trajectory loaded.")
    print(f"Number of frames: {n_frames}")
    print(f"Number of water molecules: {n_water}")

    # Show first frame info
    u.trajectory[0]
    print(f"Box dimensions: {u.trajectory.ts.dimensions}")
    print(f"First 5 O atom positions:\n{O_atoms.positions[:5]}")

    # Choose optimization method based on trajectory size
    if n_frames > 1000:
        print("Using memory-efficient method for large trajectory")
        chi4_values, t_values = calculate_chi4_memory_efficient(
            trajectory_file, a=1.0, stride=10, max_tau=500
        )
    else:
        print("Using optimized method")
        chi4_values, t_values = calculate_chi4_optimized(trajectory_file, a=1.0, stride=5)

    # Convert time to ps
    dt = 10  # ps
    t_values = t_values * dt

    # Save results (more efficient)
    results = np.column_stack((t_values, chi4_values))
    np.savetxt("chi4_values.csv", results, delimiter=",", header="t,chi4", comments="", fmt="%.6e")

    # Plot results
    plt.figure(figsize=(7, 5))
    plt.plot(
        t_values[chi4_values > 0],
        chi4_values[chi4_values > 0],
        marker="o",
        markersize=3,
        linewidth=1,
    )
    plt.xscale("log")
    plt.xlabel("Time (ps)")
    plt.ylabel(r"$\chi_4(t)$")
    plt.title("Four-point Susceptibility (Optimized)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("chi4_values_optimized.png", dpi=300, bbox_inches="tight")
    plt.show()

    print(f"Results saved to chi4_values.csv and chi4_values_optimized.png")
    print(f"Maximum chi4 value: {np.max(chi4_values):.6e}")
