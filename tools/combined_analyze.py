import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import MDAnalysis as mda
from scipy.spatial.distance import pdist, squareform
from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")


def compute_coarse_grained_coordinate(trajectory_file, method="pca", n_neighbors=6):
    """
    Compute a coarse-grained structural coordinate.

    Parameters:
    -----------
    trajectory_file : str
        Path to trajectory file
    method : str
        Method for CG coordinate ('pca', 'local_density', 'coordination', 'q6')
    n_neighbors : int
        Number of neighbors for local structure analysis

    Returns:
    --------
    xi_cg : np.ndarray
        Coarse-grained coordinate values for each frame
    """
    u = mda.Universe(trajectory_file, format="LAMMPSDUMP")
    O_atoms = u.select_atoms("type 1")
    n_frames = len(u.trajectory)
    n_atoms = len(O_atoms)

    xi_cg = np.zeros(n_frames)

    print(f"Computing CG coordinate using {method} method...")

    if method == "pca":
        # Use PCA of atomic positions as CG coordinate
        positions_all = []
        for ts in tqdm(u.trajectory, desc="Loading positions"):
            positions_all.append(O_atoms.positions.flatten())

        positions_all = np.array(positions_all)
        scaler = StandardScaler()
        positions_scaled = scaler.fit_transform(positions_all)

        pca = PCA(n_components=1)
        xi_cg = pca.fit_transform(positions_scaled).flatten()

    elif method == "local_density":
        # Local density as CG coordinate
        for i, ts in enumerate(tqdm(u.trajectory, desc="Computing local density")):
            positions = O_atoms.positions

            # Compute pairwise distances
            distances = squareform(pdist(positions))

            # Count neighbors within cutoff (e.g., 3.5 Å)
            cutoff = 3.5
            neighbors = np.sum(distances < cutoff, axis=1) - 1  # Subtract self
            xi_cg[i] = np.mean(neighbors)  # Average local coordination

    elif method == "coordination":
        # Coordination number based CG coordinate
        for i, ts in enumerate(tqdm(u.trajectory, desc="Computing coordination")):
            positions = O_atoms.positions
            box = ts.dimensions

            # Apply PBC if needed
            if box is not None:
                # Simple coordination calculation
                distances = squareform(pdist(positions))
                coordination = np.sum(distances < 3.2, axis=1) - 1
                xi_cg[i] = np.std(coordination)  # Structural disorder measure
            else:
                distances = squareform(pdist(positions))
                coordination = np.sum(distances < 3.2, axis=1) - 1
                xi_cg[i] = np.std(coordination)

    elif method == "q6":
        # Steinhardt Q6 order parameter (simplified)
        for i, ts in enumerate(tqdm(u.trajectory, desc="Computing Q6")):
            positions = O_atoms.positions

            # Simplified Q6 calculation
            distances = squareform(pdist(positions))

            q6_local = []
            for j in range(n_atoms):
                # Find nearest neighbors
                neighbors_idx = np.argsort(distances[j])[1 : n_neighbors + 1]
                neighbor_distances = distances[j][neighbors_idx]

                # Simple measure of local order
                q6_local.append(np.std(neighbor_distances))

            xi_cg[i] = np.mean(q6_local)

    return xi_cg


def compute_displacement_max_per_frame(trajectory_file, tau_frames, stride=1):
    """
    Compute maximum displacement for each frame (time origin).

    Parameters:
    -----------
    trajectory_file : str
        Path to trajectory file
    tau_frames : int
        Time lag for displacement calculation
    stride : int
        Frame stride

    Returns:
    --------
    dr_max : np.ndarray
        Maximum displacement for each valid frame
    valid_frames : np.ndarray
        Frame indices that were processed
    """
    u = mda.Universe(trajectory_file, format="LAMMPSDUMP")
    O_atoms = u.select_atoms("type 1")
    total_frames = len(u.trajectory)

    valid_frames = np.arange(0, total_frames - tau_frames, stride)
    n_valid = len(valid_frames)
    dr_max = np.zeros(n_valid)

    print(f"Computing max displacements for {n_valid} frames with tau={tau_frames}")

    for i, t0 in enumerate(tqdm(valid_frames, desc="Computing displacements")):
        # Initial positions
        u.trajectory[t0]
        r0 = O_atoms.positions.copy()

        # Final positions
        u.trajectory[t0 + tau_frames]
        r_tau = O_atoms.positions.copy()

        # Compute displacements
        displacements = np.linalg.norm(r_tau - r0, axis=1)

        # Store maximum displacement for this frame
        dr_max[i] = np.max(displacements)

    return dr_max, valid_frames


def create_2d_probability_map(x, y, bins=50, method="kde"):
    """
    Create 2D probability density map.

    Parameters:
    -----------
    x, y : np.ndarray
        Data coordinates
    bins : int or tuple
        Number of bins for histogram
    method : str
        'kde' for kernel density estimation or 'hist' for histogram

    Returns:
    --------
    X, Y : np.ndarray
        Meshgrid coordinates
    Z : np.ndarray
        Probability density values
    """
    if method == "kde":
        # Use KDE for smooth probability density
        xy = np.vstack([x, y])
        kde = gaussian_kde(xy)

        # Create grid
        x_min, x_max = np.min(x), np.max(x)
        y_min, y_max = np.min(y), np.max(y)

        x_grid = np.linspace(x_min, x_max, bins)
        y_grid = np.linspace(y_min, y_max, bins)
        X, Y = np.meshgrid(x_grid, y_grid)

        # Evaluate KDE
        positions = np.vstack([X.ravel(), Y.ravel()])
        Z = kde(positions).reshape(X.shape)

    else:
        # Use 2D histogram
        H, x_edges, y_edges = np.histogram2d(x, y, bins=bins, density=True)
        X, Y = np.meshgrid(x_edges[:-1], y_edges[:-1])
        Z = H.T

    return X, Y, Z


def plot_combined_analysis(xi_cg, ln_dr_max, title="Combined Structural-Dynamic Analysis"):
    """
    Create the combined analysis plot similar to your reference.

    Parameters:
    -----------
    xi_cg : np.ndarray
        Coarse-grained structural coordinate
    ln_dr_max : np.ndarray
        Natural log of maximum displacements
    title : str
        Plot title
    """
    # Create 2D probability density
    X, Y, Z = create_2d_probability_map(xi_cg, ln_dr_max, bins=60, method="kde")

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 8))

    # Contour plot
    levels = np.linspace(0, np.max(Z), 15)
    contour = ax.contourf(X, Y, Z, levels=levels, cmap="jet", extend="max")

    # Add contour lines
    contour_lines = ax.contour(
        X, Y, Z, levels=levels[::2], colors="black", alpha=0.3, linewidths=0.5
    )

    # Colorbar
    cbar = plt.colorbar(contour, ax=ax, shrink=0.8)
    cbar.set_label("Probability Density", fontsize=14)

    # Labels and formatting
    ax.set_xlabel(r"$\xi^{CG}$ (Å)", fontsize=16)
    ax.set_ylabel(r"ln($\Delta r_{max}$) (Å)", fontsize=16)
    ax.set_title(title, fontsize=16)

    # Set axis limits similar to your plot
    ax.set_xlim(0.4, 1.0)
    ax.set_ylim(-0.5, 2.5)

    # Grid
    ax.grid(True, alpha=0.3)

    # Adjust layout
    plt.tight_layout()

    return fig, ax


def analyze_correlation(xi_cg, dr_max):
    """
    Analyze correlation between structural and dynamic properties.
    """
    ln_dr_max = np.log(dr_max)

    # Calculate correlation coefficient
    correlation = np.corrcoef(xi_cg, ln_dr_max)[0, 1]

    # Calculate conditional averages
    n_bins = 20
    xi_bins = np.linspace(np.min(xi_cg), np.max(xi_cg), n_bins)
    conditional_avg = []
    conditional_std = []

    for i in range(len(xi_bins) - 1):
        mask = (xi_cg >= xi_bins[i]) & (xi_cg < xi_bins[i + 1])
        if np.sum(mask) > 10:  # Require at least 10 points
            conditional_avg.append(np.mean(ln_dr_max[mask]))
            conditional_std.append(np.std(ln_dr_max[mask]))
        else:
            conditional_avg.append(np.nan)
            conditional_std.append(np.nan)

    bin_centers = (xi_bins[:-1] + xi_bins[1:]) / 2

    return {
        "correlation": correlation,
        "bin_centers": bin_centers,
        "conditional_avg": np.array(conditional_avg),
        "conditional_std": np.array(conditional_std),
    }


def main():
    """Main analysis function."""

    # Configuration
    trajectory_file = "/home/debian/water/TIP4P/2005/benchmark/220/quenching/dump_H2O.lammpstrj"
    chi4_file = "quenching/chi4_values.csv"  # From previous analysis
    dt = 10  # ps per frame

    # Parameters
    cg_method = "local_density"  # Options: 'pca', 'local_density', 'coordination', 'q6'
    stride = 5  # Process every 5th frame for efficiency

    print("=== Combined Structural-Dynamic Analysis ===")

    # # Step 1: Determine optimal tau from chi4 (if available)
    # if pd.io.common.file_exists(chi4_file):
    #     chi4_data = pd.read_csv(chi4_file)
    #     max_idx = np.argmax(chi4_data["chi4"].values)
    #     tau_optimal_ps = chi4_data["t"].iloc[max_idx]
    #     tau_frames = int(tau_optimal_ps / dt)
    #     print(f"Using optimal tau from chi4: {tau_optimal_ps:.1f} ps ({tau_frames} frames)")
    # else:
    #     # Use reasonable default
    #     tau_frames = 100  # frames
    #     print(f"Using default tau: {tau_frames} frames ({tau_frames * dt} ps)")

    # # Step 2: Compute coarse-grained structural coordinate
    # xi_cg_all = compute_coarse_grained_coordinate(trajectory_file, method=cg_method)

    # # Step 3: Compute maximum displacements
    # dr_max, valid_frames = compute_displacement_max_per_frame(
    #     trajectory_file, tau_frames, stride=stride
    # )
    xi_cg_all = pd.read_csv("quenching/cg_zeta.csv", index_col=0)["cg_zeta"].values
    dr_max = pd.read_csv("quenching/max_mobility.csv", index_col=2)["displacement"].values
    tau_frames = 790  # 7900 ps
    valid_frames = len(dr_max)
    # Step 4: Align data (both arrays must have same length)
    # Take xi_cg values corresponding to valid frames
    xi_cg = xi_cg_all[:valid_frames]
    print(xi_cg.shape)
    print(f"xi_cg: {xi_cg}")
    print(f"xi_cg_all: {xi_cg_all[262794]}")
    ln_dr_max = np.log(dr_max)

    print(f"Data points: {len(xi_cg)}")
    print(f"Xi_CG range: [{np.min(xi_cg):.3f}, {np.max(xi_cg):.3f}]")
    print(f"ln(dr_max) range: [{np.min(ln_dr_max):.3f}, {np.max(ln_dr_max):.3f}]")

    # Step 5: Create the main plot
    fig, ax = plot_combined_analysis(
        xi_cg, ln_dr_max, title=f"Structural-Dynamic Correlation (τ={tau_frames*dt} ps)"
    )

    # Step 6: Analyze correlation
    correlation_analysis = analyze_correlation(xi_cg, dr_max)
    print(f"Correlation coefficient: {correlation_analysis['correlation']:.3f}")

    # Step 7: Create additional analysis plots
    fig2, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

    # Scatter plot
    ax1.scatter(xi_cg, ln_dr_max, alpha=0.5, s=10)
    ax1.set_xlabel(r"$\xi^{CG}$ (Å)")
    ax1.set_ylabel(r"ln($\Delta r_{max}$) (Å)")
    ax1.set_title(f'Scatter Plot (r={correlation_analysis["correlation"]:.3f})')
    ax1.grid(True, alpha=0.3)

    # Marginal distributions
    ax2.hist(xi_cg, bins=50, alpha=0.7, density=True, color="blue")
    ax2.set_xlabel(r"$\xi^{CG}$ (Å)")
    ax2.set_ylabel("Probability Density")
    ax2.set_title("Structural Coordinate Distribution")
    ax2.grid(True, alpha=0.3)

    ax3.hist(ln_dr_max, bins=50, alpha=0.7, density=True, color="red")
    ax3.set_xlabel(r"ln($\Delta r_{max}$) (Å)")
    ax3.set_ylabel("Probability Density")
    ax3.set_title("Dynamic Property Distribution")
    ax3.grid(True, alpha=0.3)

    # Conditional average
    valid_mask = ~np.isnan(correlation_analysis["conditional_avg"])
    bin_centers = correlation_analysis["bin_centers"][valid_mask]
    conditional_avg = correlation_analysis["conditional_avg"][valid_mask]
    conditional_std = correlation_analysis["conditional_std"][valid_mask]

    ax4.errorbar(
        bin_centers, conditional_avg, yerr=conditional_std, marker="o", capsize=3, color="green"
    )
    ax4.set_xlabel(r"$\xi^{CG}$ (Å)")
    ax4.set_ylabel(r"$\langle$ln($\Delta r_{max}$)$\rangle$ (Å)")
    ax4.set_title("Conditional Average")
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    # Step 8: Save results
    # Save data
    results_df = pd.DataFrame(
        {"frame": valid_frames, "xi_cg": xi_cg, "dr_max": dr_max, "ln_dr_max": ln_dr_max}
    )
    results_df.to_csv("combined_analysis_data.csv", index=False)

    # Save plots
    fig.savefig("combined_analysis_2d.png", dpi=300, bbox_inches="tight")
    fig2.savefig("combined_analysis_detailed.png", dpi=300, bbox_inches="tight")

    print("\nAnalysis complete!")
    print("Files saved:")
    print("- combined_analysis_data.csv")
    print("- combined_analysis_2d.png")
    print("- combined_analysis_detailed.png")

    plt.show()

    return xi_cg, dr_max, ln_dr_max


if __name__ == "__main__":
    xi_cg, dr_max, ln_dr_max = main()
