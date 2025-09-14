import MDAnalysis as mda
from MDAnalysis.analysis.hydrogenbonds.hbond_analysis import HydrogenBondAnalysis as HBA
from MDAnalysis.lib.distances import apply_PBC
from MDAnalysis.lib.nsgrid import FastNS
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm


def run_hb_analysis(dump_file, OO_cutoff=3.5, angle_cutoff=30.0, start_frame=None, end_frame=None):
    """
    Perform hydrogen bond analysis on a given trajectory file.

    Parameters:
    dump_file (str): Path to the trajectory file.
    OO_cutoff (float): Distance cutoff for O-O in Angstroms.
    angle_cutoff (float): Angle cutoff in degrees.
    start_frame (int): Starting frame for analysis.
    end_frame (int): Ending frame for analysis.

    Returns:
    list: List of hydrogen bonds identified in the trajectory.
    """
    u = mda.Universe(dump_file, format="LAMMPSDUMP")

    hbond_analysis = HBA(
        universe=u,
        donors_sel="type 1",
        hydrogens_sel="type 2",
        acceptors_sel="type 1",
        d_a_cutoff=OO_cutoff,
        d_h_a_angle_cutoff=180 - angle_cutoff,
    )

    hbond_analysis.run(start=start_frame, stop=end_frame)

    return hbond_analysis


def find_counts(group):
    donor_acceptor = pd.concat([group["donor_idx"], group["acceptor_idx"]])
    counts = donor_acceptor.value_counts().sort_index()
    return counts


def find_farthest(group):
    max_distance = group["distance"].max()
    return max_distance


if __name__ == "__main__":
    dump_file = "/home/debian/water/TIP4P/2005/benchmark/results/dump_H2O_old.lammpstrj"
    print("Running hydrogen bond analysis...")
    hbond_analysis = run_hb_analysis(dump_file)
    all_hbonds = hbond_analysis.results.hbonds
    # frame, donor_idx, hydrogen_idx, acceptor_idx, distance, angle
    pd.DataFrame(all_hbonds).to_csv(
        "hbonds_frame.csv",
        index=False,
        header=["frame", "donor_idx", "hydrogen_idx", "acceptor_idx", "distance", "angle"],
    )

    # # Using Following code
    all_hbonds = pd.read_csv(
        "hbonds_frame.csv",
        dtype={
            "frame": int,
            "donor_idx": int,
            "hydrogen_idx": int,
            "acceptor_idx": int,
            "distance": float,
            "angle": float,
        },
    ).drop(columns=["hydrogen_idx", "angle"])

    grouped = all_hbonds.groupby("frame")
    print("Calculating HB counts per oxygen...")

    max_distance_per_idx = (
        pd.concat(
            [
                all_hbonds[["frame", "donor_idx", "distance"]].rename(columns={"donor_idx": "idx"}),
                all_hbonds[["frame", "acceptor_idx", "distance"]].rename(
                    columns={"acceptor_idx": "idx"}
                ),
            ]
        )
        .groupby(["frame", "idx"])["distance"]
        .max()
        .reset_index(name="max_distance")
    )
    print(max_distance_per_idx)
    max_distance_per_idx.to_csv("max_distance_per_idx.csv", index=False)

    u = mda.Universe(dump_file, format="LAMMPSDUMP")
    O_atoms = u.select_atoms("type 1")
    global_O_indices = O_atoms.indices
    # print("global_O_indices:", global_O_indices)
    dfs = {}
    for ts in tqdm(u.trajectory, desc="Processing frames", unit="frame"):
        coords_O = O_atoms.positions
        # searcher = FastNS(cutoff=5.0, coords=coords_O, box=ts.dimensions)
        searcher = FastNS(cutoff=5.0, coords=coords_O, box=ts.dimensions, pbc=True)
        results = searcher.search(coords_O)
        pairs = results.get_pairs()
        distances = results.get_pair_distances()

        frame_hbonds = grouped.get_group(ts.frame)
        # 收集氢键对
        hb_pairs = set(zip(frame_hbonds["donor_idx"], frame_hbonds["acceptor_idx"]))
        # print(f"Frame {ts.frame}: Found {len(hb_pairs)} hydrogen bonds.")
        # print(hb_pairs)
        min_non_hb_dist = np.full(len(O_atoms), np.inf)
        for (i, j), d in zip(pairs, distances):
            if i == j:
                continue
            global_i = global_O_indices[i]
            global_j = global_O_indices[j]
            if (global_i, global_j) in hb_pairs or (global_j, global_i) in hb_pairs:
                continue
            if d < min_non_hb_dist[i]:
                min_non_hb_dist[i] = d
            if d < min_non_hb_dist[j]:
                min_non_hb_dist[j] = d

        valid_mask = min_non_hb_dist < np.inf
        valid_indices = global_O_indices[valid_mask]
        valid_distances = min_non_hb_dist[valid_mask]

        df = pd.DataFrame({"O_idx": valid_indices, "min_distance": valid_distances})
        dfs[ts.frame] = df
    all_nhb_distances = (
        pd.concat(dfs.values(), keys=dfs.keys())
        .reset_index(level=0)
        .rename(columns={"level_0": "frame"})
    )
    all_nhb_distances.to_csv("nhb_min_distances.csv", index=False)

    nhb_distances = pd.read_csv("./rst/nhb_min_distances.csv").rename(
        columns={"min_distance": "distance"}
    )

    hb_distances = pd.read_csv("./rst/max_distance_per_idx.csv").rename(
        columns={"max_distance": "distance", "idx": "O_idx"}
    )
    nhb_distances.set_index(["frame", "O_idx"], inplace=True)
    hb_distances.set_index(["frame", "O_idx"], inplace=True)
    nhb_distances["distance"] = nhb_distances["distance"] * 0.1
    hb_distances["distance"] = hb_distances["distance"] * 0.1
    combined = nhb_distances - hb_distances

    nhb_distances_old = pd.read_csv("nhb_min_distances.csv").rename(
        columns={"min_distance": "distance"}
    )
    hb_distances_old = pd.read_csv("max_distance_per_idx.csv").rename(
        columns={"max_distance": "distance", "idx": "O_idx"}
    )
    nhb_distances_old.set_index(["frame", "O_idx"], inplace=True)
    hb_distances_old.set_index(["frame", "O_idx"], inplace=True)
    nhb_distances_old["distance"] = nhb_distances_old["distance"] * 0.1
    hb_distances_old["distance"] = hb_distances_old["distance"] * 0.1
    combined_old = nhb_distances_old - hb_distances_old

    ideal = pd.read_csv("./rst/220K1bar.csv", header=None)

    plt.hist(combined["distance"], bins=400, density=True, alpha=0.5, label="New", color="blue")
    plt.hist(
        combined_old["distance"], bins=400, density=True, alpha=0.5, label="Old", color="green"
    )
    plt.plot(ideal[0], ideal[1], label="Ideal", color="orange")
    plt.xlabel("Distance Difference (Non-HB min distance - HB max distance) (nm)")
    plt.ylabel("Probability Density")
    plt.title("Distribution of Distance Differences")
    plt.legend()
    plt.savefig("distance_difference_distribution.png", dpi=300)
    plt.show()
    nhb_distances.hist(bins=100, density=True)
    plt.xlabel("Non-HB Minimum Distance (Å)")
    plt.ylabel("Probability Density")
    plt.title("Distribution of Non-HB Minimum Distances")
    plt.show()
    hb_distances.hist(bins=100, density=True)
    plt.xlabel("HB Maximum Distance (Å)")
    plt.ylabel("Probability Density")
    plt.title("Distribution of HB Maximum Distances")
    plt.show()
