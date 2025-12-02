#!/usr/bin/env python3
"""
Memory-efficient hydrogen-bond analysis (rewritten)
- Uses chunked HBA runs and appends hbonds to an HDF5 dataset (frames in ascending order)
- Avoids loading entire HDF5 into memory by scanning with a cursor
- Uses FastNS neighbor search to compute nearest non-HB distances (O(N·k) instead of O(N^2))
- Streaming I/O for counts / outputs

Usage:
    python hb_analysis_memory_efficient_rewrite.py --dump_file traj.dump --out_dir output

Requirements: MDAnalysis, h5py, numpy, matplotlib, tqdm
"""

import argparse
import os
import gc
import numpy as np
import h5py
import MDAnalysis as mda
from MDAnalysis.analysis.hydrogenbonds.hbond_analysis import HydrogenBondAnalysis as HBA
from MDAnalysis.lib.nsgrid import FastNS
from MDAnalysis.lib.distances import apply_PBC
from collections import defaultdict
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt


class HBMemoryEfficient:
    """Rewritten, more memory-efficient and faster implementation."""

    def __init__(
        self,
        dump_file,
        out_dir="output",
        chunk_size=200,
        start_frame=None,
        end_frame=None,
        mem_limit_bytes=None,
    ):
        self.dump_file = dump_file
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)

        self.chunk_size = int(chunk_size)
        self.u = mda.Universe(dump_file, format="LAMMPSDUMP")
        self.O_atoms = self.u.select_atoms("type 1")
        self.global_O_indices = self.O_atoms.indices
        self.n_atoms = len(self.O_atoms)
        self.n_frames = len(self.u.trajectory)

        self.start_frame = 0 if start_frame is None else int(start_frame)
        self.end_frame = self.n_frames if end_frame is None else int(end_frame)
        if (
            self.start_frame < 0
            or self.end_frame > self.n_frames
            or self.start_frame >= self.end_frame
        ):
            raise ValueError("Invalid start_frame or end_frame")

        # approximate memory limit (not strictly enforced, but used to pick chunk sizes)
        self.mem_limit_bytes = mem_limit_bytes

    # ------------------ stage 1: compute and write hbonds to HDF5 (ascending frames) ------------------
    def run_hb_analysis_chunked(self, OO_cutoff=3.5, angle_cutoff=30.0, hdf5_name="hbonds.h5"):
        print("[1] Running hydrogen-bond analysis (chunked) ...")
        hdf5_file = os.path.join(self.out_dir, hdf5_name)

        # dtype: frame(int32), donor(int32), hydrogen(int32), acceptor(int32), distance(float32), angle(float32)
        dtype = np.dtype(
            [
                ("frame", np.int32),
                ("donor", np.int32),
                ("hydrogen", np.int32),
                ("acceptor", np.int32),
                ("distance", np.float32),
                ("angle", np.float32),
            ]
        )

        with h5py.File(hdf5_file, "w") as h5f:
            ds = h5f.create_dataset(
                "hbonds", shape=(0,), maxshape=(None,), dtype=dtype, chunks=True, compression="gzip"
            )

            total = 0
            # Process in chunks of frames; HBA returns hbonds with absolute frame numbers
            for start in tqdm(
                range(self.start_frame, self.end_frame, self.chunk_size), desc="HBA chunks"
            ):
                stop = min(start + self.chunk_size, self.end_frame)

                # Initialize HBA for this chunk and run
                hba = HBA(
                    universe=self.u,
                    donors_sel="type 1",
                    hydrogens_sel="type 2",
                    acceptors_sel="type 1",
                    d_a_cutoff=OO_cutoff,
                    d_h_a_angle_cutoff=180.0 - angle_cutoff,
                )

                hba.run(start=start, stop=stop)

                chunk_hb = hba.results.hbonds
                if chunk_hb is None:
                    chunk_hb = np.empty((0, 6), dtype=np.float32)

                # chunk_hb shape (n,6): frame, donor, hydrogen, acceptor, distance, angle
                if len(chunk_hb) > 0:
                    # convert to structured array matching dtype
                    rec = np.empty(len(chunk_hb), dtype=dtype)
                    rec["frame"] = chunk_hb[:, 0].astype(np.int32)
                    rec["donor"] = chunk_hb[:, 1].astype(np.int32)
                    rec["hydrogen"] = chunk_hb[:, 2].astype(np.int32)
                    rec["acceptor"] = chunk_hb[:, 3].astype(np.int32)
                    rec["distance"] = chunk_hb[:, 4].astype(np.float32)
                    rec["angle"] = chunk_hb[:, 5].astype(np.float32)

                    # append preserving ascending frame order because we processed frames ascending
                    old = ds.shape[0]
                    ds.resize(old + rec.shape[0], axis=0)
                    ds[old : old + rec.shape[0]] = rec
                    total += rec.shape[0]

                # cleanup
                del hba, chunk_hb
                gc.collect()

        print(f"H-bond write complete: {total} hbonds -> {hdf5_file}")
        return hdf5_file

    # ------------------ stage 2: streaming count (scan HDF5 once) ------------------
    def process_hbond_counts_streaming(self, hdf5_file, csv_name="hb_counts_per_idx.csv"):
        print("[2] Processing hydrogen bond counts (streaming) ...")
        csv_file = os.path.join(self.out_dir, csv_name)

        # We will scan the dataset in batch windows and produce CSV streaming writes.
        with h5py.File(hdf5_file, "r") as h5f, open(csv_file, "w") as outf:
            ds = h5f["hbonds"]
            outf.write("frame,O_idx,hb_count\n")

            # accumulate counts per frame in a small dict then flush per frame
            cursor = 0
            N = ds.shape[0]
            batch = 100000  # tuneable

            # We'll iterate frames from start_frame to end_frame-1 and build counts using cursor
            for frame in tqdm(range(self.start_frame, self.end_frame), desc="frames for counts"):
                counts = defaultdict(int)

                # advance cursor to rows with this frame (dataset is in ascending frame order)
                while cursor < N:
                    row = ds[cursor]
                    rf = int(row["frame"])
                    if rf > frame:
                        break
                    if rf == frame:
                        counts[int(row["donor"])] += 1
                        counts[int(row["acceptor"])] += 1
                    cursor += 1

                # write counts for this frame
                for idx, c in counts.items():
                    outf.write(f"{frame},{idx},{c}\n")

        print(f"HB counts streaming saved to {csv_file}")
        return csv_file

    # ------------------ stage 3: compute distances frame-by-frame using FastNS and HDF5 cursor ------------------
    def calculate_distances_memory_efficient(
        self,
        hdf5_file,
        max_dist_name="max_distance_per_idx.csv",
        nhb_dist_name="nhb_min_distances.csv",
        neighbor_cutoff=6.0,
    ):
        """
        For each trajectory frame, gather hb_pairs by scanning HDF5 with a cursor (dataset ordered by frame)
        Use FastNS to find neighbors up to neighbor_cutoff and compute min non-hbond distances.
        """
        print("[3] Calculating distances (FastNS, streaming hbonds) ...")

        max_dist_file = os.path.join(self.out_dir, max_dist_name)
        nhb_dist_file = os.path.join(self.out_dir, nhb_dist_name)

        with h5py.File(hdf5_file, "r") as h5f, open(max_dist_file, "w") as fmax, open(
            nhb_dist_file, "w"
        ) as fnhb:

            ds = h5f["hbonds"]
            fmax.write("frame,idx,max_distance\n")
            fnhb.write("frame,O_idx,min_distance\n")

            # cursor across HDF5 dataset (it's important we processed hbonds in ascending frame order earlier)
            cursor = 0
            Nrows = ds.shape[0]

            # iterate trajectory frames directly from Universe (this loads one ts at a time)
            for ts in tqdm(
                self.u.trajectory[self.start_frame : self.end_frame], desc="frames distance"
            ):
                frame = ts.frame
                coords_O = self.O_atoms.positions.astype(np.float32)
                box_dims = ts.dimensions.astype(np.float32)

                # gather hb_pairs for this frame (as global indices)
                hb_pairs = set()
                max_distances = defaultdict(float)

                # advance cursor collecting rows for `frame` (dataset ordered ascending by frame)
                while cursor < Nrows:
                    row = ds[cursor]
                    rf = int(row["frame"])
                    if rf < frame:
                        cursor += 1
                        continue
                    if rf > frame:
                        break

                    donor = int(row["donor"])
                    acceptor = int(row["acceptor"])
                    dist = float(row["distance"])

                    # store as global indices (dataset used original indices from Universe)
                    hb_pairs.add((donor, acceptor))
                    hb_pairs.add((acceptor, donor))

                    # track per-atom max hb distance
                    max_distances[donor] = max(max_distances.get(donor, 0.0), dist)
                    max_distances[acceptor] = max(max_distances.get(acceptor, 0.0), dist)

                    cursor += 1

                # write max distances out (convert to consistent units if desired; here we keep raw)
                for idx, md in max_distances.items():
                    fmax.write(f"{frame},{idx},{md}\n")

                # compute min non-hbond distances using FastNS neighbor search
                if self.n_atoms == 0:
                    continue

                ns = FastNS(neighbor_cutoff, coords_O, box=box_dims)
                res = ns.self_search()
                pairs = res.get_pairs()  # (i, j) local indices; i < j normally
                pair_distances = res.get_pair_distances()

                # initialize min distances with very large value
                min_distances = np.full(self.n_atoms, np.finfo(np.float32).max, dtype=np.float32)

                # iterate neighbor pairs
                for (i, j), d in zip(pairs, pair_distances):

                    gi = int(self.global_O_indices[i])
                    gj = int(self.global_O_indices[j])

                    # skip if hb pair
                    if (gi, gj) in hb_pairs:
                        continue

                    # update local minima
                    if d < min_distances[i]:
                        min_distances[i] = d
                    if d < min_distances[j]:
                        min_distances[j] = d

                # write non-hbond minima
                for local_i, val in enumerate(min_distances):
                    if val < np.finfo(np.float32).max:
                        global_idx = int(self.global_O_indices[local_i])
                        fnhb.write(f"{frame},{global_idx},{val}\n")

                # periodic cleanup
                if frame % 100 == 0:
                    gc.collect()

        print("Distance calculations complete")
        return max_dist_file, nhb_dist_file

    # ------------------ stage 4: compute zeta streaming ------------------
    def calculate_zeta_streaming(self, max_dist_file, nhb_dist_file, zeta_name="zeta.csv"):
        print("[4] Calculating zeta (streaming) ...")
        zeta_file = os.path.join(self.out_dir, zeta_name)

        # load max distances into a dict keyed by (frame, idx) - this can be large but usually smaller than hbonds
        maxd = {}
        with open(max_dist_file, "r") as f:
            next(f)
            for line in f:
                frame, idx, md = line.strip().split(",")
                maxd[(int(frame), int(idx))] = float(md)

        with open(nhb_dist_file, "r") as fin, open(zeta_file, "w") as fout:
            fout.write("frame,O_idx,zeta\n")
            next(fin)
            for line in fin:
                frame, oidx, mind = line.strip().split(",")
                key = (int(frame), int(oidx))
                md = maxd.get(key, 0.0)
                zeta = float(mind) - float(md)
                fout.write(f"{frame},{oidx},{zeta}\n")

        print(f"Zeta saved to {zeta_file}")
        return zeta_file

    # ------------------ plotting (unchanged streaming-friendly) ------------------
    def plot_hb_distribution(self, hb_counts_file):
        print("[5] Plotting distributions ...")
        hb_counts = []
        with open(hb_counts_file, "r") as f:
            next(f)
            for line in f:
                _, _, c = line.strip().split(",")
                hb_counts.append(int(c))

        if len(hb_counts) == 0:
            print("No hb count data to plot")
            return

        plt.figure(figsize=(8, 5))
        plt.hist(
            hb_counts, bins=range(0, max(hb_counts) + 2), density=True, align="left", rwidth=0.8
        )
        plt.xlabel("Number of Hydrogen Bonds per Oxygen")
        plt.ylabel("Probability Density")
        plt.title("Distribution of Hydrogen Bonds per Oxygen")
        plt.grid(alpha=0.3)
        plt.text(
            0.95,
            0.95,
            f"Average: {np.mean(hb_counts):.2f}",
            transform=plt.gca().transAxes,
            ha="right",
            va="top",
        )
        plt.savefig(
            os.path.join(self.out_dir, "hb_count_distribution.png"), dpi=300, bbox_inches="tight"
        )
        plt.close()

    # ------------------ orchestrator ------------------
    def run_complete(self, OO_cutoff=3.5, angle_cutoff=30.0, neighbor_cutoff=6.0):
        hb_h5 = self.run_hb_analysis_chunked(
            OO_cutoff=OO_cutoff, angle_cutoff=angle_cutoff, hdf5_name="hbonds.h5"
        )
        counts_csv = self.process_hbond_counts_streaming(hb_h5)
        max_csv, nhb_csv = self.calculate_distances_memory_efficient(
            hb_h5, neighbor_cutoff=neighbor_cutoff
        )
        zeta_csv = self.calculate_zeta_streaming(max_csv, nhb_csv)
        self.plot_hb_distribution(counts_csv)

        # optionally remove hb hdf5 to save space
        # try:
        #     os.remove(hb_h5)
        # except Exception:
        #     pass

        return {
            "hb_counts": counts_csv,
            "max_distances": max_csv,
            "nhb_distances": nhb_csv,
            "zeta": zeta_csv,
        }


# ------------------ CLI ------------------
def main():
    parser = argparse.ArgumentParser(
        description="Memory-efficient hydrogen bond analysis (rewritten)"
    )
    parser.add_argument("--dump_file", required=True)
    parser.add_argument("--out_dir", default="output")
    parser.add_argument("--chunk_size", type=int, default=200)
    parser.add_argument("--start_frame", type=int, default=None)
    parser.add_argument("--end_frame", type=int, default=None)
    parser.add_argument(
        "--neighbor_cutoff", type=float, default=6.0, help="FastNS neighbor cutoff in angstrom"
    )

    args = parser.parse_args()

    analyzer = HBMemoryEfficient(
        dump_file=args.dump_file,
        out_dir=args.out_dir,
        chunk_size=args.chunk_size,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
    )

    results = analyzer.run_complete(
        OO_cutoff=3.5, angle_cutoff=30.0, neighbor_cutoff=args.neighbor_cutoff
    )
    print("Results:")
    for k, v in results.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
