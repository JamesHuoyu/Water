import MDAnalysis as mda
from MDAnalysis.analysis.hydrogenbonds.hbond_analysis import HydrogenBondAnalysis as HBA
from MDAnalysis.lib.distances import apply_PBC
from MDAnalysis.lib.nsgrid import FastNS
from MDAnalysis.analysis import rdf
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt


def fix_cg_zeta_calculation():
    """
    Fixed version of CG zeta calculation with proper PBC handling.
    """
    dump_file = "/home/debian/water/TIP4P/2005/benchmark/220/quenching/dump_H2O.lammpstrj"

    print("Running hydrogen bond analysis...")

    # Load distance data
    nhb_distances = pd.read_csv("quenching/nhb_min_distances.csv").rename(
        columns={"min_distance": "distance"}
    )

    hb_distances = pd.read_csv("quenching/max_distance_per_idx.csv").rename(
        columns={"max_distance": "distance", "idx": "O_idx"}
    )

    print(f"NHB distances shape: {nhb_distances.shape}")
    print(f"HB distances shape: {hb_distances.shape}")
    print(f"NHB columns: {nhb_distances.columns.tolist()}")
    print(f"HB columns: {hb_distances.columns.tolist()}")

    # Convert units (assuming original is in nm, convert to Å)
    # NOTE: Remove duplicate conversion - you were doing *0.1 twice
    nhb_distances["distance"] = nhb_distances["distance"]  # Keep original units
    hb_distances["distance"] = hb_distances["distance"]  # Keep original units

    # Ensure both DataFrames have the same structure
    # Method 1: Merge approach (recommended)
    print("Merging distance data...")

    # Merge on common columns (frame and O_idx)
    merged_data = pd.merge(
        nhb_distances,
        hb_distances,
        on=["frame", "O_idx"],
        suffixes=("_nhb", "_hb"),
        how="inner",  # Only keep rows that exist in both datasets
    )

    # Calculate zeta as difference
    merged_data["zeta"] = merged_data["distance_nhb"] - merged_data["distance_hb"]

    # Save the zeta data with proper structure
    zeta = merged_data[["frame", "O_idx", "zeta"]].copy()
    zeta.to_csv("quenching/zeta.csv", index=False)

    print(f"Zeta data shape: {zeta.shape}")
    print(f"Zeta range: [{zeta['zeta'].min():.4f}, {zeta['zeta'].max():.4f}]")

    # Load trajectory for neighbor analysis
    u = mda.Universe(dump_file, format="LAMMPSDUMP")
    O_atoms = u.select_atoms("type 1")
    global_O_indices = O_atoms.indices

    print(f"Total oxygen atoms: {len(O_atoms)}")

    # Compute RDF for cutoff determination
    print("Computing RDF...")
    rdf_analyzer = rdf.InterRDF(O_atoms, O_atoms, range=(1.0, 6.0), nbins=200)
    rdf_analyzer.run()

    rdf_profile = rdf_analyzer.results.rdf
    bins = rdf_analyzer.results.bins

    peak_index = np.argmax(rdf_profile)
    print(f"First peak at r = {bins[peak_index]:.3f} Å with g(r) = {rdf_profile[peak_index]:.3f}")

    # Save RDF
    df_rdf = pd.DataFrame({"r": bins, "rdf_OO_mean": rdf_profile})
    df_rdf.set_index("r", inplace=True)
    df_rdf.to_csv("quenching/rdf_OO.csv", index=True)

    # Find first minimum after peak for cutoff
    min_index = peak_index + np.argmin(rdf_profile[peak_index : peak_index + 50])
    rmin = bins[min_index]

    print(f"Using cutoff radius from RDF first minimum: {rmin:.3f} Å")

    # Create lookup dictionary for faster zeta access
    print("Creating zeta lookup dictionary...")
    zeta_lookup = {}
    for _, row in zeta.iterrows():
        frame = int(row["frame"])
        o_idx = int(row["O_idx"])
        zeta_val = row["zeta"]

        if frame not in zeta_lookup:
            zeta_lookup[frame] = {}
        zeta_lookup[frame][o_idx] = zeta_val

    # Calculate coarse-grained zeta
    print("Computing CG zeta with proper PBC...")
    cg_zeta_data = []

    # Process trajectory with progress bar
    for ts in tqdm(u.trajectory, desc="Processing frames", unit="frame"):
        frame = ts.frame
        coords_O = O_atoms.positions
        box = ts.dimensions

        # Use FastNS with proper PBC
        searcher = FastNS(cutoff=rmin, coords=coords_O, box=box, pbc=True)
        results = searcher.search(coords_O)
        pairs = results.get_pairs()
        distances = results.get_pair_distances()

        # Build neighbors dictionary
        neighbors_dict = {i: [] for i in range(len(O_atoms))}
        for (i, j), d in zip(pairs, distances):
            if i != j:  # Skip self-pairs
                neighbors_dict[i].append(j)

        # Calculate CG zeta for each atom
        for i, neighbors in neighbors_dict.items():
            global_i = global_O_indices[i]

            # Get zeta values for neighbors
            neighbor_zeta_values = []

            for j in neighbors:
                global_j = global_O_indices[j]

                # Efficient lookup using dictionary
                if frame in zeta_lookup and global_j in zeta_lookup[frame]:
                    zeta_val = zeta_lookup[frame][global_j]
                    if not np.isnan(zeta_val):  # Skip NaN values
                        neighbor_zeta_values.append(zeta_val)

            # Calculate CG zeta
            if len(neighbor_zeta_values) > 0:
                cg_zeta = np.mean(neighbor_zeta_values)
            else:
                # Handle case with no valid neighbors
                # Option 1: Use own zeta value if available
                if frame in zeta_lookup and global_i in zeta_lookup[frame]:
                    cg_zeta = zeta_lookup[frame][global_i]
                else:
                    # Option 2: Use global mean or set to NaN
                    cg_zeta = np.nan

            cg_zeta_data.append(
                {
                    "frame": frame,
                    "O_idx": global_i,
                    "cg_zeta": cg_zeta,
                    "n_neighbors": len(neighbor_zeta_values),
                }
            )

    # Create and save CG zeta DataFrame
    cg_zeta_df = pd.DataFrame(cg_zeta_data)
    print("Finishing CG zeta computation...")

    # Statistics
    n_total = len(cg_zeta_df)
    n_nan = cg_zeta_df["cg_zeta"].isna().sum()
    n_valid = n_total - n_nan

    print(f"CG Zeta Statistics:")
    print(f"Total entries: {n_total:,}")
    print(f"Valid entries: {n_valid:,} ({100*n_valid/n_total:.1f}%)")
    print(f"NaN entries: {n_nan:,} ({100*n_nan/n_total:.1f}%)")
    print(f"Average neighbors per atom: {cg_zeta_df['n_neighbors'].mean():.1f}")

    # Save results
    cg_zeta_df.to_csv("quenching/cg_zeta.csv", index=False)

    return zeta, cg_zeta_df


def plot_results():
    """
    Plot the results with proper handling of NaN values.
    """
    print("Creating plots...")

    # Load data
    zeta = pd.read_csv("quenching/zeta.csv")
    cg_zeta_df = pd.read_csv("quenching/cg_zeta.csv")

    # Load ideal distribution if available
    try:
        ideal = pd.read_csv("./rst/220K1bar.csv", header=None)
        has_ideal = True
    except FileNotFoundError:
        print("Ideal distribution file not found, skipping...")
        has_ideal = False

    # Create plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

    # Main distribution plot
    ax1.hist(
        zeta["zeta"],
        bins=200,
        density=True,
        alpha=0.6,
        label="ζ (individual)",
        color="blue",
        edgecolor="black",
        linewidth=0.5,
    )

    # Remove NaN values for histogram
    cg_zeta_valid = cg_zeta_df["cg_zeta"].dropna()
    ax1.hist(
        cg_zeta_valid,
        bins=200,
        density=True,
        alpha=0.6,
        label=f"ζ_cg (coarse-grained)",
        color="green",
        edgecolor="black",
        linewidth=0.5,
    )

    # if has_ideal:
    #     ax1.plot(ideal[0], ideal[1], label="Ideal", color="orange", linewidth=2)

    ax1.set_xlabel(r"$\zeta$ and $\zeta_{cg}$ (Å)")
    ax1.set_ylabel("Probability Density")
    ax1.set_title(r"Distribution of $\zeta$")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Statistics comparison
    zeta_stats = {
        "mean": zeta["zeta"].mean(),
        "std": zeta["zeta"].std(),
        "min": zeta["zeta"].min(),
        "max": zeta["zeta"].max(),
    }

    cg_stats = {
        "mean": cg_zeta_valid.mean(),
        "std": cg_zeta_valid.std(),
        "min": cg_zeta_valid.min(),
        "max": cg_zeta_valid.max(),
    }

    # Box plot comparison
    data_to_plot = [zeta["zeta"].values, cg_zeta_valid.values]
    ax2.boxplot(data_to_plot, labels=["ζ", "ζ_cg"])
    ax2.set_ylabel(r"$\zeta$ values (Å)")
    ax2.set_title("Statistical Comparison")
    ax2.grid(True, alpha=0.3)

    # Scatter plot (sample for correlation)
    if len(cg_zeta_valid) > 1000:
        sample_size = 1000
        sample_idx = np.random.choice(len(cg_zeta_valid), sample_size, replace=False)
        x_sample = zeta["zeta"].iloc[sample_idx]
        y_sample = cg_zeta_valid.iloc[sample_idx]
    else:
        x_sample = zeta["zeta"][: len(cg_zeta_valid)]
        y_sample = cg_zeta_valid

    ax3.scatter(x_sample, y_sample, alpha=0.5, s=10)
    ax3.set_xlabel(r"$\zeta$ (individual)")
    ax3.set_ylabel(r"$\zeta_{cg}$ (coarse-grained)")
    ax3.set_title("Individual vs Coarse-grained")

    # Calculate and display correlation
    if len(x_sample) == len(y_sample) and len(x_sample) > 1:
        correlation = np.corrcoef(x_sample, y_sample)[0, 1]
        ax3.text(
            0.05,
            0.95,
            f"r = {correlation:.3f}",
            transform=ax3.transAxes,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

    ax3.grid(True, alpha=0.3)

    # Statistics table
    ax4.axis("off")
    stats_text = f"""
    Statistics Summary:
    
    Individual ζ:
    Mean: {zeta_stats['mean']:.4f} Å
    Std:  {zeta_stats['std']:.4f} Å
    Range: [{zeta_stats['min']:.4f}, {zeta_stats['max']:.4f}]
    Count: {len(zeta):,}
    
    Coarse-grained ζ_cg:
    Mean: {cg_stats['mean']:.4f} Å
    Std:  {cg_stats['std']:.4f} Å
    Range: [{cg_stats['min']:.4f}, {cg_stats['max']:.4f}]
    Count: {len(cg_zeta_valid):,}
    
    NaN values: {len(cg_zeta_df) - len(cg_zeta_valid):,}
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
    plt.savefig("quenching/zeta_cg_fixed.png", dpi=300, bbox_inches="tight")
    plt.show()

    # Print summary
    print(f"\nSummary:")
    print(f"Individual ζ: {zeta_stats['mean']:.4f} ± {zeta_stats['std']:.4f} Å")
    print(f"Coarse-grained ζ_cg: {cg_stats['mean']:.4f} ± {cg_stats['std']:.4f} Å")


if __name__ == "__main__":
    # Run the fixed calculation
    zeta, cg_zeta_df = fix_cg_zeta_calculation()

    # Create plots
    plot_results()

    print("\nFixed analysis complete!")
    print("Files saved:")
    print("- quenching/zeta.csv (corrected)")
    print("- quenching/cg_zeta.csv (with neighbor counts)")
    print("- quenching/rdf_OO.csv")
    print("- quenching/zeta_cg_fixed.png")
