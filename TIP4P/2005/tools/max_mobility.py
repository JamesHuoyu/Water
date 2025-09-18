import MDAnalysis as mda
from MDAnalysis.lib.distances import apply_PBC
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm


def compute_mobility_at_tau(trajectory_file, tau_frames, use_pbc=True, stride=1, max_origins=None):
    """
    Compute mobility (displacement) of water molecules at a specific time lag tau.

    Parameters:
    -----------
    trajectory_file : str
        Path to the trajectory file in LAMMPSDUMP format.
    tau_frames : int
        Time lag in frames for displacement calculation.
    use_pbc : bool
        Whether to apply periodic boundary conditions.
    stride : int
        Process every nth frame to reduce computation.
    max_origins : int or None
        Maximum number of time origins to consider.

    Returns:
    --------
    displacements : np.ndarray
        Array of shape (n_origins, n_water) containing displacements.
    valid_origins : int
        Number of valid time origins processed.
    """
    u = mda.Universe(trajectory_file, format="LAMMPSDUMP")
    O_atoms = u.select_atoms("type 1")
    n_water = len(O_atoms)

    if n_water == 0:
        raise ValueError("No water molecules found in the trajectory.")

    total_frames = len(u.trajectory)
    available_origins = total_frames - tau_frames

    if available_origins <= 0:
        raise ValueError(f"Tau ({tau_frames}) is too large. Maximum tau: {total_frames - 1}")

    # Determine frame indices to process
    frame_indices = np.arange(0, available_origins, stride)
    if max_origins:
        frame_indices = frame_indices[:max_origins]

    n_origins = len(frame_indices)
    print(f"Processing {n_origins} time origins with tau = {tau_frames} frames")

    displacements = np.zeros((n_origins, n_water))

    for i, t0 in enumerate(tqdm(frame_indices, desc="Computing displacements")):
        # Get initial positions
        u.trajectory[t0]
        r0 = O_atoms.positions.copy()
        box0 = u.trajectory.ts.dimensions if use_pbc else None

        # Get final positions
        u.trajectory[t0 + tau_frames]
        r_tau = O_atoms.positions.copy()

        # Calculate displacement
        displacement_vectors = r_tau - r0

        if use_pbc and box0 is not None:
            # Apply periodic boundary conditions
            displacement_vectors = apply_PBC(displacement_vectors, box0)

        # Calculate displacement magnitudes
        displacements[i] = np.linalg.norm(displacement_vectors, axis=1)

    return displacements, n_origins


def compute_mobility_distribution(trajectory_file, tau_frames, bins=100, use_pbc=True):
    """
    Compute and analyze the distribution of mobilities at a given tau.

    Parameters:
    -----------
    trajectory_file : str
        Path to trajectory file.
    tau_frames : int
        Time lag in frames.
    bins : int
        Number of histogram bins.
    use_pbc : bool
        Whether to apply periodic boundary conditions.

    Returns:
    --------
    hist_data : dict
        Dictionary containing histogram data and statistics.
    """
    displacements, n_origins = compute_mobility_at_tau(trajectory_file, tau_frames, use_pbc=use_pbc)

    all_displacements = displacements.flatten()

    # Calculate statistics
    stats = {
        "mean": np.mean(all_displacements),
        "std": np.std(all_displacements),
        "median": np.median(all_displacements),
        "q25": np.percentile(all_displacements, 25),
        "q75": np.percentile(all_displacements, 75),
        "q90": np.percentile(all_displacements, 90),
        "q95": np.percentile(all_displacements, 95),
        "q99": np.percentile(all_displacements, 99),
        "max": np.max(all_displacements),
        "n_origins": n_origins,
        "n_molecules": len(displacements[0]) if len(displacements) > 0 else 0,
        "total_samples": len(all_displacements),
    }

    # Create histogram
    hist, bin_edges = np.histogram(all_displacements, bins=bins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    return {
        "displacements": all_displacements,
        "histogram": hist,
        "bin_centers": bin_centers,
        "bin_edges": bin_edges,
        "stats": stats,
    }


def plot_mobility_analysis(hist_data, tau_frames, dt=10, save_prefix="mobility"):
    """
    Create comprehensive plots for mobility analysis.
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

    displacements = hist_data["displacements"]
    stats = hist_data["stats"]
    tau_ps = tau_frames * dt

    # Histogram
    ax1.hist(displacements, bins=100, density=True, alpha=0.7, color="skyblue", edgecolor="black")
    ax1.axvline(stats["mean"], color="red", linestyle="--", label=f"Mean: {stats['mean']:.2f} Å")
    ax1.axvline(
        stats["median"], color="orange", linestyle="--", label=f"Median: {stats['median']:.2f} Å"
    )
    ax1.set_xlabel("Displacement (Å)")
    ax1.set_ylabel("Probability Density")
    ax1.set_title(f"Displacement Distribution at τ = {tau_ps} ps")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Log-scale histogram
    ax2.hist(
        displacements, bins=100, density=True, alpha=0.7, color="lightgreen", edgecolor="black"
    )
    ax2.set_xlabel("Displacement (Å)")
    ax2.set_ylabel("Probability Density")
    ax2.set_yscale("log")
    ax2.set_title(f"Log-scale Distribution")
    ax2.grid(True, alpha=0.3)

    # Cumulative distribution
    sorted_disp = np.sort(displacements)
    cumulative = np.arange(1, len(sorted_disp) + 1) / len(sorted_disp)
    ax3.plot(sorted_disp, cumulative, "b-", linewidth=2)
    ax3.axhline(0.95, color="red", linestyle="--", label=f"95th percentile: {stats['q95']:.2f} Å")
    ax3.axhline(
        0.99, color="orange", linestyle="--", label=f"99th percentile: {stats['q99']:.2f} Å"
    )
    ax3.set_xlabel("Displacement (Å)")
    ax3.set_ylabel("Cumulative Probability")
    ax3.set_title("Cumulative Distribution")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Statistics table
    ax4.axis("off")
    stats_text = f"""
    Statistics for τ = {tau_ps} ps
    
    Mean: {stats['mean']:.3f} Å
    Std:  {stats['std']:.3f} Å
    Median: {stats['median']:.3f} Å
    
    Percentiles:
    25th: {stats['q25']:.3f} Å
    75th: {stats['q75']:.3f} Å
    90th: {stats['q90']:.3f} Å
    95th: {stats['q95']:.3f} Å
    99th: {stats['q99']:.3f} Å
    Max:  {stats['max']:.3f} Å
    
    Samples: {stats['total_samples']:,}
    Origins: {stats['n_origins']:,}
    Molecules: {stats['n_molecules']:,}
    """
    ax4.text(
        0.1,
        0.9,
        stats_text,
        transform=ax4.transAxes,
        fontsize=10,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="lightgray", alpha=0.8),
    )

    plt.tight_layout()
    plt.savefig(f"{save_prefix}_analysis.png", dpi=300, bbox_inches="tight")
    plt.show()

    return fig


if __name__ == "__main__":
    dump_file = "/home/debian/water/TIP4P/2005/benchmark/220/quenching/dump_H2O.lammpstrj"
    t_alpha = 7900  # 7900 ps
    dt = 10  # 10 ps
    frame_alpha = t_alpha // dt
    displacements, n_origins = compute_mobility_at_tau(dump_file, frame_alpha, use_pbc=False)
    all_displacements = displacements.flatten()

    n_molecules = displacements.shape[1]
    molecule_ids = np.tile(np.arange(n_molecules), n_origins)
    origin_ids = np.repeat(np.arange(n_origins), n_molecules)

    df = pd.DataFrame(
        {"displacement": all_displacements, "molecule_id": molecule_ids, "origin_frame": origin_ids}
    )
    df.to_csv("quenching/max_mobility.csv", index=False)

    # Analyze distribution
    hist_data = compute_mobility_distribution(dump_file, frame_alpha, bins=100)
    plot_mobility_analysis(hist_data, frame_alpha, dt, "quenching/max_mobility")
    # Print statistics
    stats = hist_data["stats"]
    print(f"\nMobility Statistics at tau = {t_alpha} ps:")
    print(f"Mean displacement: {stats['mean']:.3f} Å")
    print(f"95th percentile: {stats['q95']:.3f} Å")
    print(f"Maximum displacement: {stats['max']:.3f} Å")
    print(f"Total samples: {stats['total_samples']:,}")
