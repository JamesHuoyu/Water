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


class NeighborAnalysis:
    def __init__(
        self, dump_file, out_dir="output", chunk_size=100, start_frame=None, end_frame=None
    ):
        self.dump_file = dump_file
        self.out_dir = out_dir
        self.chunk_size = chunk_size
        self.u = mda.Universe(dump_file, format="LAMMPSDUMP")
        self.O_atoms = self.u.select_atoms("type 1")
        self.global_O_indices = self.O_atoms.indices
        self.n_frames = len(self.u.trajectory)

        self.start_frame = start_frame if start_frame is not None else 0
        self.end_frame = end_frame if end_frame is not None else self.n_frames
        if (
            self.start_frame < 0
            or self.end_frame > self.n_frames
            or self.start_frame >= self.end_frame
        ):
            raise ValueError("Invalid start_frame or end_frame")
        # 确保输出目录存在
        os.makedirs(out_dir, exist_ok=True)

    def get_four_neighbors(self, box):
        positions = self.O_atoms.positions
        dist_matrix = distance_array(positions, positions, box=box)
        np.fill_diagonal(dist_matrix, np.inf)
        neighbor_indices = np.argsort(dist_matrix, axis=1)[:, :4]
        neighbor_distances = np.sort(dist_matrix, axis=1)[:, :4]
        return neighbor_indices, neighbor_distances

    def chunked_neighbor_analysis(self):
        out_file = os.path.join(self.out_dir, "four_neighbors_distances.csv")
        with open(out_file, "w") as f:
            f.write("frame,O_idx,neighbor_rank,neighbor_O_idx,distance,ts_box\n")
        print("Beginning chunked neighbors analysis...")
        for start in tqdm(range(self.start_frame, self.end_frame, self.chunk_size), desc="Chunks"):
            end = min(start + self.chunk_size, self.end_frame)
            chunk_data = []
            for frame in range(start, end):
                self.u.trajectory[frame]
                box = self.u.trajectory[frame].dimensions
                neighbor_indices, neighbor_distances = self.get_four_neighbors(box)
                for i, O_idx in enumerate(self.global_O_indices):
                    for n in range(4):
                        chunk_data.append(
                            [
                                frame,
                                O_idx,
                                n + 1,
                                self.global_O_indices[neighbor_indices[i, n]],
                                neighbor_distances[i, n],
                                box,
                            ]
                        )
            chunk_df = pd.DataFrame(
                chunk_data,
                columns=["frame", "O_idx", "neighbor_rank", "neighbor_O_idx", "distance", "ts_box"],
            )
            chunk_df.to_csv(out_file, mode="a", header=False, index=False)
            del chunk_df, chunk_data  # 释放内存
            gc.collect()
        print(f"Neighbor analysis complete. Results saved to {out_file}")

    def compute_q_chunked(self, df_path=None, chunk_size=1000, frame_chunk_size=10):
        """
        Memory-efficient computation of tetrahedral order parameter Q using chunking.

        Parameters:
        -----------
        df_path : str or None
            Path to the neighbors distance CSV file
        chunk_size : int
            Number of atoms to process at once
        frame_chunk_size : int
            Number of frames to process at once
        """
        if df_path is None:
            df_path = os.path.join(self.out_dir, "four_neighbors_distances.csv")

        print("Computing tetrahedral order parameter Q (chunked)...")

        # Check if file exists and get basic info
        if not os.path.exists(df_path):
            raise FileNotFoundError(f"Neighbors file not found: {df_path}")

        # Read file info without loading all data
        print("Analyzing input file...")
        total_rows = sum(1 for _ in open(df_path)) - 1  # Subtract header
        print(f"Total rows in input: {total_rows:,}")

        # Get unique frames and atoms for memory estimation
        sample_df = pd.read_csv(df_path, nrows=10000)  # Sample first 10k rows
        unique_frames = sample_df["frame"].nunique()
        unique_atoms = sample_df["O_idx"].nunique()
        print(f"Estimated frames: {unique_frames}, atoms: {unique_atoms}")

        # Calculate processing parameters
        total_frames = self.end_frame - self.start_frame
        frames_per_chunk = min(frame_chunk_size, total_frames)
        atoms_per_chunk = min(chunk_size, len(self.global_O_indices))

        print(f"Processing {total_frames} frames in chunks of {frames_per_chunk}")
        print(f"Processing {len(self.global_O_indices)} atoms in chunks of {atoms_per_chunk}")

        # Output file setup
        out_file = os.path.join(self.out_dir, "tetrahedral_order_parameter_Q.csv")

        # Initialize output file with header
        with open(out_file, "w") as f:
            f.write("frame,O_idx,Q\n")

        total_q_values = 0

        # Process in frame chunks
        for frame_start in tqdm(
            range(self.start_frame, self.end_frame, frames_per_chunk), desc="Frame chunks"
        ):
            frame_end = min(frame_start + frames_per_chunk, self.end_frame)
            frame_list = list(range(frame_start, frame_end))

            print(f"Processing frames {frame_start} to {frame_end-1}")

            # Load data for current frame chunk
            try:
                # Read only relevant frames to reduce memory
                chunk_df = pd.read_csv(
                    df_path,
                    dtype={
                        "frame": int,
                        "O_idx": int,
                        "neighbor_O_idx": int,
                        "neighbor_rank": int,
                        "distance": float,
                    },
                )

                # Filter for current frames
                chunk_df = chunk_df[chunk_df["frame"].isin(frame_list)]

                if chunk_df.empty:
                    print(f"No data found for frames {frame_start}-{frame_end-1}")
                    continue

                print(f"Loaded {len(chunk_df):,} rows for frames {frame_start}-{frame_end-1}")

            except Exception as e:
                print(f"Error loading data for frames {frame_start}-{frame_end-1}: {e}")
                continue

            # Process each frame in the chunk
            frame_q_values = []

            for frame in frame_list:
                if frame % 50 == 0:  # Progress update
                    print(f"  Processing frame {frame}")

                frame_data = chunk_df[chunk_df["frame"] == frame]

                if frame_data.empty:
                    continue

                # Load trajectory frame for positions
                try:
                    self.u.trajectory[frame]
                    current_positions = self.O_atoms.positions.copy()
                    current_box = self.u.trajectory.ts.dimensions
                except Exception as e:
                    print(f"Error loading trajectory frame {frame}: {e}")
                    continue

                # Process atoms in chunks for this frame
                frame_atoms = frame_data["O_idx"].unique()

                for atom_start_idx in range(0, len(frame_atoms), atoms_per_chunk):
                    atom_end_idx = min(atom_start_idx + atoms_per_chunk, len(frame_atoms))
                    atom_chunk = frame_atoms[atom_start_idx:atom_end_idx]

                    # Process each atom in the current chunk
                    for O_idx in atom_chunk:
                        # Skip if not in our global indices
                        if O_idx not in self.global_O_indices:
                            continue

                        # Get neighbors for this atom
                        neighbors = frame_data[frame_data["O_idx"] == O_idx].sort_values(
                            "neighbor_rank"
                        )

                        if len(neighbors) < 4:
                            # Not enough neighbors for tetrahedral calculation
                            continue

                        # Get central atom position
                        try:
                            central_atom_mask = self.O_atoms.indices == O_idx
                            if not np.any(central_atom_mask):
                                continue

                            O_pos = current_positions[central_atom_mask][0]

                            # Get neighbor positions and calculate vectors
                            r_vectors = []

                            for _, row in neighbors.head(
                                4
                            ).iterrows():  # Take only first 4 neighbors
                                neighbor_idx = row["neighbor_O_idx"]

                                # Find neighbor position
                                neighbor_mask = self.O_atoms.indices == neighbor_idx
                                if not np.any(neighbor_mask):
                                    continue

                                neighbor_pos = current_positions[neighbor_mask][0]

                                # Apply minimum image convention
                                vec = neighbor_pos - O_pos
                                if current_box is not None:
                                    vec = apply_PBC(vec.reshape(1, -1), current_box)[0]

                                # Normalize vector
                                vec_norm = np.linalg.norm(vec)
                                if vec_norm > 0:
                                    r_vectors.append(vec / vec_norm)

                            # Calculate Q parameter if we have exactly 4 vectors
                            if len(r_vectors) == 4:
                                r_vectors = np.array(r_vectors)

                                # Calculate dot products between all pairs
                                cos_theta = np.dot(r_vectors, r_vectors.T)
                                cos_theta = np.clip(cos_theta, -1.0, 1.0)

                                # Get upper triangular elements (excluding diagonal)
                                triu_indices = np.triu_indices(4, k=1)
                                cos_triu = cos_theta[triu_indices]

                                # Calculate tetrahedral order parameter
                                # Q = 1 - (3/8) * sum((cos(θ_jk) + 1/3)²) for all j<k pairs
                                q = 1 - (3 / 8) * np.sum((cos_triu + 1 / 3) ** 2)

                                frame_q_values.append({"frame": frame, "O_idx": O_idx, "Q": q})

                        except Exception as e:
                            print(f"Error processing atom {O_idx} in frame {frame}: {e}")
                            continue

            # Save results for this frame chunk
            if frame_q_values:
                chunk_results_df = pd.DataFrame(frame_q_values)

                # Append to output file
                chunk_results_df.to_csv(out_file, mode="a", header=False, index=False)

                total_q_values += len(frame_q_values)
                print(
                    f"  Saved {len(frame_q_values)} Q values for frames {frame_start}-{frame_end-1}"
                )

            # Clean up memory
            del chunk_df, frame_q_values
            if "chunk_results_df" in locals():
                del chunk_results_df
            gc.collect()

        print(f"Q computation complete. Total {total_q_values} values saved to {out_file}")

    def compute_q_streaming(self, df_path=None, batch_size=10000):
        """
        Alternative streaming approach that processes the CSV in batches.
        Even more memory efficient for very large datasets.
        """
        if df_path is None:
            df_path = os.path.join(self.out_dir, "four_neighbors_distances.csv")

        print("Computing tetrahedral order parameter Q (streaming)...")

        out_file = os.path.join(self.out_dir, "tetrahedral_order_parameter_Q.csv")

        # Initialize output file
        with open(out_file, "w") as f:
            f.write("frame,O_idx,Q\n")

        total_processed = 0
        current_frame = None
        frame_data_buffer = []

        # Read CSV in chunks
        chunk_iter = pd.read_csv(
            df_path,
            chunksize=batch_size,
            dtype={
                "frame": int,
                "O_idx": int,
                "neighbor_O_idx": int,
                "neighbor_rank": int,
                "distance": float,
            },
        )

        for chunk_num, chunk in enumerate(tqdm(chunk_iter, desc="Processing CSV chunks")):

            # Filter for our frame range
            chunk = chunk[(chunk["frame"] >= self.start_frame) & (chunk["frame"] < self.end_frame)]

            if chunk.empty:
                continue

            # Group by frame and process
            for frame_id, frame_group in chunk.groupby("frame"):

                if frame_id != current_frame:
                    # Process previous frame if exists
                    if frame_data_buffer:
                        q_results = self._process_frame_q(frame_data_buffer, current_frame)
                        if q_results:
                            # Save results
                            results_df = pd.DataFrame(q_results)
                            results_df.to_csv(out_file, mode="a", header=False, index=False)
                            total_processed += len(q_results)

                    # Start new frame
                    current_frame = frame_id
                    frame_data_buffer = []

                # Add to buffer
                frame_data_buffer.extend(frame_group.to_dict("records"))

            if chunk_num % 10 == 0:  # Progress update
                print(
                    f"Processed {chunk_num * batch_size:,} rows, {total_processed} Q values calculated"
                )

        # Process final frame
        if frame_data_buffer and current_frame is not None:
            q_results = self._process_frame_q(frame_data_buffer, current_frame)
            if q_results:
                results_df = pd.DataFrame(q_results)
                results_df.to_csv(out_file, mode="a", header=False, index=False)
                total_processed += len(q_results)

        print(f"Streaming Q computation complete. {total_processed} values saved to {out_file}")
        return (
            pd.read_csv(out_file) if os.path.getsize(out_file) > 50 else None
        )  # Only load if file has data

    def _process_frame_q(self, frame_data, frame_id):
        """
        Helper function to process Q calculation for a single frame.
        """
        try:
            # Load trajectory frame
            self.u.trajectory[frame_id]
            current_positions = self.O_atoms.positions.copy()
            current_box = self.u.trajectory.ts.dimensions
            # print(f"Processing frame {frame_id} with box {current_box}")  # Debug info
        except:
            return []

        # Convert to DataFrame for easier processing
        frame_df = pd.DataFrame(frame_data)
        q_results = []

        # Process each unique atom
        for O_idx in frame_df["O_idx"].unique():
            if O_idx not in self.global_O_indices:
                continue

            neighbors = frame_df[frame_df["O_idx"] == O_idx].sort_values("neighbor_rank")

            if len(neighbors) < 4:
                continue

            try:
                # Get positions and calculate Q
                central_mask = self.O_atoms.indices == O_idx
                if not np.any(central_mask):
                    continue

                O_pos = current_positions[central_mask][0]
                # O_pos = apply_PBC(O_pos.reshape(1, -1), current_box)[0]
                r_vectors = []

                for _, row in neighbors.head(4).iterrows():
                    neighbor_mask = self.O_atoms.indices == row["neighbor_O_idx"]
                    if not np.any(neighbor_mask):
                        continue
                    neighbor_pos = current_positions[neighbor_mask][0]
                    # neighbor_pos = apply_PBC(neighbor_pos.reshape(1, -1), current_box)[0]
                    # print(f"Neighbor pos: {neighbor_pos}")  # Debug info
                    # print(f"O pos: {O_pos}")  # Debug info
                    vec = neighbor_pos - O_pos
                    box_length = current_box[:3]
                    vec -= box_length * np.round(vec / box_length)
                    vec_norm = np.linalg.norm(vec)
                    if vec_norm > 0:
                        r_vectors.append(vec / vec_norm)

                if len(r_vectors) == 4:
                    r_vectors = np.array(r_vectors)
                    cos_theta = np.dot(r_vectors, r_vectors.T)
                    cos_theta = np.clip(cos_theta, -1.0, 1.0)
                    triu_indices = np.triu_indices(4, k=1)
                    cos_triu = cos_theta[triu_indices]
                    q = 1 - (3 / 8) * np.sum((cos_triu + 1 / 3) ** 2)
                    q_results.append({"frame": frame_id, "O_idx": O_idx, "Q": q})

            except Exception as e:
                continue

        return q_results


if __name__ == "__main__":
    dump_file = "/home/debian/water/TIP4P/2005/traj_2.5e-5_246.lammpstrj"
    out_dir = "/home/debian/water/TIP4P/2005/2020/rst"
    chunk_size = 100  # Adjust based on memory capacity
    start_frame = 2000
    end_frame = None  # Process all frames

    analyzer = NeighborAnalysis(
        dump_file=dump_file,
        out_dir=out_dir,
        chunk_size=chunk_size,
        start_frame=start_frame,
        end_frame=end_frame,
    )
    analyzer.chunked_neighbor_analysis()
    analyzer.compute_q_streaming(None, 5000)  # Use default path for distances CSV
