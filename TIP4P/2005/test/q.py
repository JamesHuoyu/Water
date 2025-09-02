import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
import glob
import os
from tqdm import tqdm


def compute_q(positions):
    N = len(positions)
    q_list = []
    dist_mat = distance_matrix(positions, positions)

    for i in range(N):
        neighbors = np.argsort(dist_mat[i])[1:5]  # Get indices of 4 nearest neighbors
        cos_angles = []
        for j in range(3):
            for k in range(j + 1, 4):
                vec1 = positions[neighbors[j]] - positions[i]
                vec2 = positions[neighbors[k]] - positions[i]
                cos_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
                cos_angles.append((cos_angle + 1 / 3) ** 2)  # (cos(theta) + 1/3)^2
        q = 1 - (3 / 8) * sum(cos_angles)
        q_list.append(q)
    return np.array(q_list)


def load_dump_to_array(filename):
    frames = []
    ids_ref = None
    box_bounds = []
    with open(filename) as f:
        while True:
            line = f.readline()
            if not line:
                break
            if "ITEM: TIMESTEP" not in line:
                continue
            _ = f.readline()  # timestep value
            f.readline()  # ITEM: NUMBER OF ATOMS
            n = int(f.readline())
            f.readline()  # ITEM: BOX BOUNDS ...
            for _ in range(3):
                bounds = list(map(float, f.readline().split()))
                box_bounds.append(bounds)
            Lx = box_bounds[-3][1] - box_bounds[-3][0]
            Ly = box_bounds[-2][1] - box_bounds[-2][0]
            Lz = box_bounds[-1][1] - box_bounds[-1][0]
            header = f.readline().strip()  # ITEM: ATOMS id xu yu zu
            cols = header.split()[2:]
            assert cols[:4] == ["id", "xu", "yu", "zu"]
            data = np.loadtxt([f.readline() for _ in range(n)])
            ids = data[:, 0].astype(int)
            # xyz = data[:, 1:4] * np.array([Lx, Ly, Lz])  # Scale to box size
            xyz = data[:, 1:4] - data[:, 1:4] // np.array([Lx, Ly, Lz]) * np.array(
                [Lx, Ly, Lz]
            )  # Use fractional coordinates directly
            if ids_ref is None:
                order = np.argsort(ids)
                ids_ref = ids[order]
            else:
                order = np.argsort(ids)
                assert np.all(ids[order] == ids_ref)
            frames.append(xyz[order])
    X = np.stack(frames, axis=0)  # (T,N,3)
    return X


X = load_dump_to_array("/home/debian/water/TIP4P/2005/test/dump_O_250MPa.lammpstrj")
q_values_list = []
for frame in tqdm(X[:2001], desc="Computing q values for each frame"):
    q_values = compute_q(frame)
    q_values_list.append(q_values)

q_values_list = np.array(q_values_list)  # Shape (T, N)
q_values = q_values_list.flatten()  # Flatten to 1D array
plt.hist(q_values, bins=500, histtype="step", density=True)
plt.xlabel("q")
plt.ylabel("Probability Density")
plt.title("Distribution of Tetrahedral Order Parameter q")
# plt.savefig("q_distribution.png")
plt.show()
