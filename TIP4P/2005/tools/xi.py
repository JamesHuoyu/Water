import MDAnalysis as mda
from MDAnalysis.analysis.hydrogenbonds.hbond_analysis import HydrogenBondAnalysis as HBA
from MDAnalysis.lib.distances import distance_array
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm


def run_hb_analysis(dump_file, OO_cutoff=3.5, angle_cutoff=30.0, start_frame=None, end_frame=None):
    """运行氢键分析，返回 MDAnalysis HBA 对象"""
    u = mda.Universe(dump_file, format="LAMMPSDUMP")
    hbond_analysis = HBA(
        universe=u,
        donors_sel="type 1",  # O 作为 donor
        hydrogens_sel="type 2",  # H
        acceptors_sel="type 1",  # O 作为 acceptor
        d_a_cutoff=OO_cutoff,
        d_h_a_angle_cutoff=180 - angle_cutoff,
    )
    hbond_analysis.run(start=start_frame, stop=end_frame)
    return hbond_analysis


if __name__ == "__main__":
    dump_file = "/home/debian/water/TIP4P/2005/benchmark/results/dump_H2O_old.lammpstrj"

    print("Running hydrogen bond analysis...")
    hbond_analysis = run_hb_analysis(dump_file)
    all_hbonds = hbond_analysis.results.hbonds
    hbonds_df = pd.DataFrame(
        all_hbonds,
        columns=["frame", "donor_idx", "hydrogen_idx", "acceptor_idx", "distance", "angle"],
    ).drop(columns=["hydrogen_idx", "angle"])

    grouped = hbonds_df.groupby("frame")

    # Universe for distance计算
    u = mda.Universe(dump_file, format="LAMMPSDUMP")
    O_atoms = u.select_atoms("type 1")
    O_indices = O_atoms.indices

    z_values = []

    print("Processing frames for Tanaka z ...")
    for ts in tqdm(u.trajectory, unit="frame"):
        # 获取该帧的氢键对
        if ts.frame not in grouped.groups:
            continue
        frame_hbonds = grouped.get_group(ts.frame)
        hb_pairs = set(zip(frame_hbonds["donor_idx"], frame_hbonds["acceptor_idx"]))

        coords = O_atoms.positions
        dmat = distance_array(coords, coords, box=ts.dimensions)

        for i, gi in enumerate(O_indices):
            dists = dmat[i, :]
            order = np.argsort(dists)

            last_hb_dist = None
            first_nonhb_dist = None

            for j in order:
                if i == j:
                    continue
                gj = O_indices[j]
                is_hb = (gi, gj) in hb_pairs or (gj, gi) in hb_pairs
                if is_hb:
                    last_hb_dist = dists[j]
                else:
                    if first_nonhb_dist is None:
                        first_nonhb_dist = dists[j]

            if last_hb_dist is not None and first_nonhb_dist is not None:
                # 转 nm
                z_values.append((first_nonhb_dist - last_hb_dist) * 0.1)

    z_values = np.array(z_values)
    print("总共得到 ζ 样本数:", len(z_values))

    # 保存
    pd.DataFrame({"z": z_values}).to_csv("rst/original_distribution.csv", index=False)

    # 绘制分布
    plt.hist(z_values, bins=100, density=True, alpha=0.6, color="blue", label="This work")
    ideal = pd.read_csv("./rst/220K1bar.csv", header=None)
    plt.plot(ideal[0], ideal[1], label="Ideal (Tanaka)", color="orange")
    plt.axvline(0, color="red", linestyle="--", label="ζ=0")
    plt.xlabel("ζ = d_nonHB(first) - d_HB(last) (nm)")
    plt.ylabel("Probability Density")
    plt.legend()
    plt.title("Tanaka ζ Distribution")
    plt.show()
