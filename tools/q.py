"""
Tetrahedral Order Parameter Calculator
Optimized using MDAnalysis with efficient neighbor search and vectorized operations.
"""

import warnings

warnings.filterwarnings("ignore", message="Reader has no dt information, set to 1.0 ps")

import MDAnalysis as mda
from MDAnalysis.lib.distances import apply_PBC, distance_array
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import gc


class TetrahedralOrderAnalysis:
    """
    Compute tetrahedral order parameter Q for water molecules.

    The tetrahedral order parameter Q is calculated as:
    Q = 1 - (3/8) * sum((cos(θ_jk) + 1/3)²) for all j<k pairs

    where θ_jk is the angle between vectors from central atom to neighbor j and k.

    For perfect tetrahedral geometry, Q = 1
    """

    def __init__(
        self,
        dump_file,
        out_dir="output",
        n_neighbors=4,
        start_frame=None,
        end_frame=None,
        neighbor_cutoff=3.5,  # Angstrom - typical H-bond distance
    ):
        """
        Initialize the tetrahedral order analyzer.

        Parameters:
        -----------
        dump_file : str
            Path to LAMMPS dump file
        out_dir : str
            Output directory for results
        n_neighbors : int
            Number of nearest neighbors to consider (default 4 for tetrahedral)
        start_frame : int or None
            Starting frame number
        end_frame : int or None
            Ending frame number
        neighbor_cutoff : float
            Cutoff distance for neighbor search in Angstrom
        """
        self.dump_file = dump_file
        self.out_dir = out_dir
        self.n_neighbors = n_neighbors
        self.neighbor_cutoff = neighbor_cutoff

        # Load universe
        self.u = mda.Universe(dump_file, format="LAMMPSDUMP")

        # Select oxygen atoms (assuming type 1 based on original code)
        self.O_atoms = self.u.select_atoms("type 1")

        if len(self.O_atoms) == 0:
            raise ValueError("No oxygen atoms found. Check atom type selection.")

        # Store global indices for reference
        self.global_O_indices = self.O_atoms.indices
        self.n_atoms = len(self.O_atoms)

        # Frame range
        self.n_frames = len(self.u.trajectory)
        self.start_frame = start_frame if start_frame is not None else 0
        self.end_frame = end_frame if end_frame is not None else self.n_frames

        # Validate frame range
        if (
            self.start_frame < 0
            or self.end_frame > self.n_frames
            or self.start_frame >= self.end_frame
        ):
            raise ValueError(
                f"Invalid frame range: {self.start_frame} to {self.end_frame} (total frames: {self.n_frames})"
            )

        # Ensure output directory exists
        os.makedirs(out_dir, exist_ok=True)

        print(f"Initialized analysis:")
        print(f"  Trajectory: {dump_file}")
        print(f"  Oxygen atoms: {self.n_atoms}")
        print(
            f"  Frames: {self.start_frame} to {self.end_frame-1} ({self.end_frame - self.start_frame} frames)"
        )

    def _find_neighbors_vectorized(self, positions, box):
        """
        Find n nearest neighbors using vectorized operations.

        Parameters:
        -----------
        positions : numpy.ndarray
            Shape (n_atoms, 3) array of atomic positions
        box : numpy.ndarray
            Simulation box dimensions

        Returns:
        --------
        neighbor_indices : numpy.ndarray
            Shape (n_atoms, n_neighbors) array of neighbor indices
        neighbor_distances : numpy.ndarray
            Shape (n_atoms, n_neighbors) array of neighbor distances
        """
        # Compute full distance matrix
        dist_matrix = distance_array(positions, positions, box=box)

        # Set diagonal to infinity to exclude self
        np.fill_diagonal(dist_matrix, np.inf)

        # Find n nearest neighbors
        neighbor_indices = np.argsort(dist_matrix, axis=1)[:, : self.n_neighbors]
        neighbor_distances = np.take_along_axis(dist_matrix, neighbor_indices, axis=1)

        return neighbor_indices, neighbor_distances

    def _compute_q_vectorized(self, positions, neighbor_indices, box):
        """
        Compute tetrahedral order parameter Q using vectorized operations.

        Parameters:
        -----------
        positions : numpy.ndarray
            Shape (n_atoms, 3) array of atomic positions
        neighbor_indices : numpy.ndarray
            Shape (n_atoms, n_neighbors) array of neighbor indices
        box : numpy.ndarray or None
            Simulation box dimensions (None for non-periodic)

        Returns:
        --------
        q_values : numpy.ndarray
            Shape (n_atoms,) array of Q values
        """
        n_atoms = len(positions)
        n_neighbors = neighbor_indices.shape[1]

        if n_neighbors != 4:
            raise ValueError(f"Expected 4 neighbors, got {n_neighbors}")

        # Initialize Q values (set to NaN for atoms with insufficient neighbors)
        q_values = np.full(n_atoms, np.nan)

        # Get central atom positions and neighbor positions
        central_positions = positions  # Shape: (n_atoms, 3)
        neighbor_positions = positions[neighbor_indices]  # Shape: (n_atoms, n_neighbors, 3)

        # Compute vectors from central atoms to neighbors
        # Shape: (n_atoms, n_neighbors, 3)
        r_vectors = neighbor_positions - central_positions[:, np.newaxis, :]

        # Apply minimum image convention for periodic systems
        if box is not None:
            # Reshape for apply_PBC: (n_atoms * n_neighbors, 3)
            r_flat = r_vectors.reshape(-1, 3)
            r_flat_pbc = apply_PBC(r_flat, box)
            r_vectors = r_flat_pbc.reshape(n_atoms, n_neighbors, 3)

        # Normalize vectors
        vec_norms = np.linalg.norm(r_vectors, axis=2, keepdims=True)
        vec_norms[vec_norms == 0] = 1  # Avoid division by zero
        r_vectors_normalized = r_vectors / vec_norms

        # Compute cosine of angles between all pairs of vectors
        # Shape: (n_atoms, n_neighbors, n_neighbors)
        cos_theta = np.einsum("ijk,ilk->ijl", r_vectors_normalized, r_vectors_normalized)

        # Clip to valid range [-1, 1] to avoid numerical errors
        cos_theta = np.clip(cos_theta, -1.0, 1.0)

        # Extract upper triangular elements (excluding diagonal)
        # For 4 neighbors, there are 6 pairs: (0,1), (0,2), (0,3), (1,2), (1,3), (2,3)
        triu_indices = np.triu_indices(n_neighbors, k=1)
        cos_pairs = cos_theta[:, triu_indices[0], triu_indices[1]]  # Shape: (n_atoms, 6)

        # Compute tetrahedral order parameter
        # Q = 1 - (3/8) * sum((cos(θ_jk) + 1/3)²)
        q_values = 1 - (3.0 / 8.0) * np.sum((cos_pairs + 1.0 / 3.0) ** 2, axis=1)

        return q_values

    def compute_q_direct(self, save_neighbors=True):
        """
        Compute tetrahedral order parameter Q directly from trajectory.
        This is the most efficient method as it avoids intermediate files.

        Parameters:
        -----------
        save_neighbors : bool
            Whether to save neighbor information to CSV

        Returns:
        --------
        q_df : pandas.DataFrame
            DataFrame with columns: frame, O_idx, Q
        """
        out_q_file = os.path.join(self.out_dir, "tetrahedral_order_parameter_Q.csv")
        out_neighbors_file = os.path.join(self.out_dir, "four_neighbors_distances.csv")

        # Initialize output files
        with open(out_q_file, "w") as f:
            f.write("frame,O_idx,Q\n")

        if save_neighbors:
            with open(out_neighbors_file, "w") as f:
                f.write("frame,O_idx,neighbor_rank,neighbor_O_idx,distance\n")

        all_q_data = []

        print("Computing tetrahedral order parameter Q (direct method)...")

        for frame in tqdm(range(self.start_frame, self.end_frame), desc="Processing frames"):
            # Load frame
            self.u.trajectory[frame]
            positions = self.O_atoms.positions.copy()
            box = self.u.trajectory.ts.dimensions

            # Find neighbors
            neighbor_indices, neighbor_distances = self._find_neighbors_vectorized(positions, box)

            # Compute Q parameter
            q_values = self._compute_q_vectorized(positions, neighbor_indices, box)

            # Collect results
            frame_data = []
            for i, O_idx in enumerate(self.global_O_indices):
                frame_data.append({"frame": frame, "O_idx": O_idx, "Q": q_values[i]})

                # Save neighbor info if requested
                if save_neighbors:
                    with open(out_neighbors_file, "a") as f:
                        for n in range(self.n_neighbors):
                            neighbor_idx = self.global_O_indices[neighbor_indices[i, n]]
                            dist = neighbor_distances[i, n]
                            f.write(f"{frame},{O_idx},{n+1},{neighbor_idx},{dist}\n")

            all_q_data.extend(frame_data)

            # Save results periodically (every 100 frames)
            if (frame - self.start_frame + 1) % 100 == 0:
                temp_df = pd.DataFrame(frame_data)
                temp_df.to_csv(out_q_file, mode="a", header=False, index=False)
                all_q_data = []  # Clear buffer

        # Save remaining data
        if all_q_data:
            pd.DataFrame(all_q_data).to_csv(out_q_file, mode="a", header=False, index=False)

        print(f"Q computation complete. Results saved to {out_q_file}")
        if save_neighbors:
            print(f"Neighbor information saved to {out_neighbors_file}")

        # Return results
        return pd.read_csv(out_q_file)

    def compute_q_from_neighbors(self, neighbors_file=None):
        """
        Compute Q parameter from pre-computed neighbor file.
        Useful if you want to recompute Q with different parameters.

        Parameters:
        -----------
        neighbors_file : str or None
            Path to neighbors CSV file

        Returns:
        --------
        q_df : pandas.DataFrame
            DataFrame with columns: frame, O_idx, Q
        """
        if neighbors_file is None:
            neighbors_file = os.path.join(self.out_dir, "four_neighbors_distances.csv")

        if not os.path.exists(neighbors_file):
            raise FileNotFoundError(f"Neighbors file not found: {neighbors_file}")

        out_q_file = os.path.join(self.out_dir, "tetrahedral_order_parameter_Q.csv")

        # Initialize output
        with open(out_q_file, "w") as f:
            f.write("frame,O_idx,Q\n")

        print("Computing Q from neighbors file...")

        # Read neighbors file in chunks
        all_q_data = []
        current_frame = None
        frame_neighbors = {}

        chunk_iter = pd.read_csv(
            neighbors_file,
            chunksize=10000,
            dtype={
                "frame": int,
                "O_idx": int,
                "neighbor_O_idx": int,
                "neighbor_rank": int,
                "distance": float,
            },
        )

        for chunk in tqdm(chunk_iter, desc="Processing neighbor chunks"):
            # Process each frame in the chunk
            for frame_id, frame_group in chunk.groupby("frame"):
                if frame_id < self.start_frame or frame_id >= self.end_frame:
                    continue

                # If we've completed processing a frame, compute Q
                if current_frame is not None and frame_id != current_frame:
                    q_results = self._compute_q_for_frame(current_frame, frame_neighbors)
                    all_q_data.extend(q_results)
                    frame_neighbors = {}

                current_frame = frame_id

                # Accumulate neighbors for current frame
                for O_idx in frame_group["O_idx"].unique():
                    atom_neighbors = frame_group[frame_group["O_idx"] == O_idx]
                    frame_neighbors[O_idx] = atom_neighbors

        # Process last frame
        if current_frame is not None:
            q_results = self._compute_q_for_frame(current_frame, frame_neighbors)
            all_q_data.extend(q_results)

        # Save results
        if all_q_data:
            pd.DataFrame(all_q_data).to_csv(out_q_file, mode="a", header=False, index=False)

        print(f"Q computation complete. Results saved to {out_q_file}")

        return pd.read_csv(out_q_file)

    def _compute_q_for_frame(self, frame_id, frame_neighbors):
        """
        Compute Q for a single frame using neighbor information.

        Parameters:
        -----------
        frame_id : int
            Frame number
        frame_neighbors : dict
            Dictionary mapping O_idx to neighbor DataFrame

        Returns:
        --------
        results : list
            List of dictionaries with frame, O_idx, Q
        """
        # Load frame
        self.u.trajectory[frame_id]
        positions = self.O_atoms.positions.copy()
        box = self.u.trajectory.ts.dimensions

        results = []

        for O_idx, neighbors in frame_neighbors.items():
            if len(neighbors) < 4:
                continue

            # Get neighbor indices
            neighbors_sorted = neighbors.sort_values("neighbor_rank").head(4)
            neighbor_indices = [
                self._get_local_idx(n_idx) for n_idx in neighbors_sorted["neighbor_O_idx"]
            ]

            if None in neighbor_indices:
                continue

            # Get positions
            central_pos = positions[self._get_local_idx(O_idx)]
            neighbor_pos = positions[neighbor_indices]

            # Compute Q
            neighbor_indices_array = np.array([neighbor_indices])
            q_values = self._compute_q_vectorized(
                np.array([central_pos]), neighbor_indices_array, box
            )

            results.append({"frame": frame_id, "O_idx": O_idx, "Q": q_values[0]})

        return results

    def _get_local_idx(self, global_idx):
        """Get local index from global index."""
        try:
            return np.where(self.global_O_indices == global_idx)[0][0]
        except IndexError:
            return None

    def analyze_statistics(self, q_df=None):
        """
        Compute statistics of Q parameter distribution.

        Parameters:
        -----------
        q_df : pandas.DataFrame or None
            DataFrame with Q values. If None, load from file.

        Returns:
        --------
        stats : dict
            Dictionary of statistics
        """
        if q_df is None:
            q_file = os.path.join(self.out_dir, "tetrahedral_order_parameter_Q.csv")
            if not os.path.exists(q_file):
                raise FileNotFoundError(f"Q file not found: {q_file}")
            q_df = pd.read_csv(q_file)

        # Filter out NaN values
        valid_q = q_df["Q"].dropna()

        stats = {
            "mean": valid_q.mean(),
            "std": valid_q.std(),
            "min": valid_q.min(),
            "max": valid_q.max(),
            "median": valid_q.median(),
            "percentile_25": valid_q.quantile(0.25),
            "percentile_75": valid_q.quantile(0.75),
        }

        # Print statistics
        print("\n=== Tetrahedral Order Parameter Statistics ===")
        print(f"Total Q values: {len(valid_q)}")
        print(f"Mean:  {stats['mean']:.4f} ± {stats['std']:.4f}")
        print(f"Median: {stats['median']:.4f}")
        print(f"Range:  [{stats['min']:.4f}, {stats['max']:.4f}]")
        print(f"25th-75th percentile: [{stats['percentile_25']:.4f}, {stats['percentile_75']:.4f}]")

        # Interpretation
        if stats["mean"] > 0.8:
            print("Interpretation: High tetrahedral order (ice-like)")
        elif stats["mean"] > 0.5:
            print("Interpretation: Moderate tetrahedral order (liquid water)")
        else:
            print("Interpretation: Low tetrahedral order (disordered)")

        return stats

    def visualize_q_distribution(self, q_df=None, save_path=None):
        """
        Visualize Q parameter distribution.

        Parameters:
        -----------
        q_df : pandas.DataFrame or None
            DataFrame with Q values
        save_path : str or None
            Path to save figure
        """
        if q_df is None:
            q_file = os.path.join(self.out_dir, "tetrahedral_order_parameter_Q.csv")
            if not os.path.exists(q_file):
                raise FileNotFoundError(f"Q file not found: {q_file}")
            q_df = pd.read_csv(q_file)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Overall distribution
        axes[0].hist(q_df["Q"].dropna(), bins=50, edgecolor="black", alpha=0.7)
        axes[0].set_xlabel("Tetrahedral Order Parameter Q")
        axes[0].set_ylabel("Frequency")
        axes[0].set_title("Overall Q Distribution")
        axes[0].axvline(
            q_df["Q"].mean(), color="red", linestyle="--", label=f'Mean: {q_df["Q"].mean():.3f}'
        )
        axes[0].legend()

        # Q vs Frame
        frame_stats = q_df.groupby("frame")["Q"].agg(["mean", "std"]).reset_index()
        axes[1].errorbar(
            frame_stats["frame"],
            frame_stats["mean"],
            yerr=frame_stats["std"],
            fmt="o",
            capsize=3,
            alpha=0.6,
        )
        axes[1].set_xlabel("Frame")
        axes[1].set_ylabel("Q")
        axes[1].set_title("Q vs Frame")
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path is None:
            save_path = os.path.join(self.out_dir, "q_distribution.png")

        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to {save_path}")
        plt.close()

        return fig


if __name__ == "__main__":
    # Configuration
    dump_file = "TIP4P/Ice/test/traj_5e-6_225_100000.lammpstrj"
    out_dir = "/home/debian/water/TIP4P/Ice/test/5e-6"
    start_frame = 0
    end_frame = None  # Process all frames

    # Initialize analyzer
    analyzer = TetrahedralOrderAnalysis(
        dump_file=dump_file,
        out_dir=out_dir,
        start_frame=start_frame,
        end_frame=end_frame,
    )

    # Compute Q parameter (most efficient method)
    q_df = analyzer.compute_q_direct(save_neighbors=True)

    # Analyze statistics
    stats = analyzer.analyze_statistics(q_df)

    # Visualize results
    analyzer.visualize_q_distribution(q_df)
