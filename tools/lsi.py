import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
from tqdm import tqdm


def compute_lsi(positions, cutoff=3.7):
    N = len(positions)
    lsi_list = []
    dist_mat = distance_matrix(positions, positions)
    n_neigbors_list = []

    for i in range(N):

        all_dicts = np.sort(dist_mat[i][dist_mat[i] > 0])  # Exclude self-distance (0)
        n_i = np.sum(all_dicts < cutoff)
        n_neigbors_list.append(n_i)

        if n_i < 1:
            lsi_list.append(np.nan)
            continue

        r_list = all_dicts[: n_i + 1]
        delta = np.diff(r_list)
        mean_delta = np.mean(delta)
        lsi = np.mean((delta - mean_delta) ** 2)
        lsi_list.append(lsi)

    return np.array(lsi_list), n_neigbors_list


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
lsi_values = []
n_nerigbor_list = []
n_nerigbor_list_all = []
for frame in tqdm(X, desc="Computing LSI"):
    lsi, n_nerigbor_list = compute_lsi(frame)
    lsi_values.append(lsi)
    n_nerigbor_list_all.extend(n_nerigbor_list)

lsi_values = np.array(lsi_values)
lsi = lsi_values.flatten()
n_nerigbor_list_all = np.array(n_nerigbor_list_all).flatten()
plt.hist(
    n_nerigbor_list_all,
    bins=np.arange(0, 20) - 0.5,
    density=True,
    edgecolor="black",
    facecolor="none",
)
plt.xlabel("Number of Neighbors within 3.7 Å")
plt.ylabel("Probability Density")
plt.title("Histogram of Number of Neighbors")
# plt.savefig("n_neighbors_histogram.png", dpi=300)
plt.show()
plt.hist(lsi, bins=500, histtype="step", density=True)
plt.xlabel("LSI")
plt.ylabel("Probability Density")
plt.title("Histogram of Local Structure Index (LSI)")
# plt.savefig("lsi_histogram.png", dpi=300)
plt.show()
