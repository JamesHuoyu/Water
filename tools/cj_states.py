import h5py
from collections import defaultdict
import numpy as np
from tqdm import tqdm
from MDAnalysis.lib.distances import minimize_vectors


def load_hbond_neighbors(h5file):
    """
    neighbors[i][t] = set(j)
    """
    neighbors = defaultdict(lambda: defaultdict(set))

    with h5py.File(h5file, "r") as f:
        hb = f["hbonds"]
        for rec in hb:
            t = int(rec["frame"])
            d = int(rec["donor"] // 3)
            a = int(rec["acceptor"] // 3)
            neighbors[d][t].add(a)
            neighbors[a][t].add(d)

    return neighbors


def identify_CJ_states(neighbors_i, n_frames):
    """
    对单个分子 i:
    返回 [{'C':(t0,t1), 'J':(t1,t2)}, ...]
    """
    segments = []
    t = 0

    while t < n_frames:
        # 必须从 4 配位开始
        if len(neighbors_i.get(t, [])) < 4:
            t += 1
            continue

        C_neighbors = set(neighbors_i[t])
        tC0 = t

        # C state：至少一个原始 H-bond 仍存在
        while t < n_frames:
            if any(j in neighbors_i.get(t, []) for j in C_neighbors):
                t += 1
            else:
                break
        tC1 = t

        # J state：直到形成“完全不同的 4 配位”
        while t < n_frames:
            neigh = neighbors_i.get(t, set())
            if len(neigh) == 4 and neigh.isdisjoint(C_neighbors):
                break
            t += 1
        tJ1 = t

        if tJ1 > tC1:
            segments.append({"C": (tC0, tC1), "J": (tC1, tJ1)})

    return segments


def analyze_jumps(CJ_segments, coords, boxes, mol_id, dt):
    tau_C = []
    tau_J = []
    rJ = []

    total_J_frames = 0
    total_frames = coords.shape[0]

    for seg in CJ_segments:
        tC0, tC1 = seg["C"]
        tJ0, tJ1 = seg["J"]

        tau_C.append((tC1 - tC0) * dt)
        tau_J.append((tJ1 - tJ0) * dt)

        if tJ1 > tJ0:
            dr = coords[tJ1, mol_id] - coords[tJ0, mol_id]
            dr = minimize_vectors(dr, boxes[tJ0])
            rJ.append(np.linalg.norm(dr))

            total_J_frames += tJ1 - tJ0

    rho_J = total_J_frames / total_frames

    return {"tau_C": np.array(tau_C), "tau_J": np.array(tau_J), "rJ": np.array(rJ), "rho_J": rho_J}


def compute_JMSD(rJ, max_jump=20):
    """
    JMSD(ΘJ)
    """
    JMSD = []
    for Θ in range(1, max_jump + 1):
        if len(rJ) < Θ:
            break
        JMSD.append(np.mean(np.sum(rJ[:Θ] ** 2)))
    return np.array(JMSD)


def run_pipeline(hbond_h5, coords, boxes, dt):
    print("Begin Loading")
    neighbors = load_hbond_neighbors(hbond_h5)
    print("Loading Finished!")
    n_frames, n_mols, _ = coords.shape
    n_frames -= 1

    all_tau_C = []
    all_tau_J = []
    all_rJ = []
    rho_J_list = []

    for i in tqdm(range(n_mols), desc="Processing molecules"):
        CJ_segments = identify_CJ_states(neighbors[i], n_frames)
        results = analyze_jumps(CJ_segments, coords, boxes, i, dt)

        all_tau_C.extend(results["tau_C"])
        all_tau_J.extend(results["tau_J"])
        all_rJ.extend(results["rJ"])
        rho_J_list.append(results["rho_J"])
    return {
        "tau_C": np.array(all_tau_C),
        "tau_J": np.array(all_tau_J),
        "rJ": np.array(all_rJ),
        "rho_J": np.mean(rho_J_list),
        "JMSD": compute_JMSD(np.array(all_rJ)),
    }
