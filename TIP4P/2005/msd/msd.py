# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.stats import linregress


# def load_msd(fname, timestep=0.001):
#     data = np.loadtxt(fname)
#     time_ps = data[:, 0] * timestep
#     msd = data[:, 1]
#     return time_ps, msd


# def adaptive_window(t):
#     if t < 0.1:
#         return 40
#     elif t < 10:
#         return 20
#     elif t < 100:
#         return 5
#     else:
#         return 3


# def compute_local_slope(time, msd):
#     log_t = np.log10(time)
#     log_msd = np.log10(msd)
#     slope = np.zeros_like(log_t)
#     for i in range(len(log_t)):
#         window = adaptive_window(time[i])
#         i1 = max(0, i - window)
#         i2 = min(len(log_t), i + window)
#         x = log_t[i1:i2]
#         y = log_msd[i1:i2]
#         if len(x) > 2:
#             slope[i] = np.polyfit(x, y, 1)[0]
#         else:
#             slope[i] = np.nan
#     return slope


# def fit_diffusive_region(time, msd, t_min=500, t_max=2000):
#     mask = (time > t_min) & (time < t_max)
#     slope, intercept, _, _, _ = linregress(time[mask], msd[mask])
#     D = slope / 6.0
#     return D, slope, intercept


# def plot_msd_analysis(time, msd, slope, D, t_min, t_max, filename):
#     fig, ax1 = plt.subplots(figsize=(8, 5))
#     ax1.plot(time, msd, label="MSD", color="blue")
#     ax1.set_xlabel("Time (ps)")
#     ax1.set_ylabel("MSD (Å²)", color="blue")
#     ax1.tick_params(axis="y", labelcolor="blue")

#     ax2 = ax1.twinx()
#     ax2.plot(time, slope, label="Local slope", color="red", alpha=0.6)
#     ax2.set_ylabel("Local slope (d log(MSD) / d log(t))", color="red")
#     ax2.tick_params(axis="y", labelcolor="red")

#     # 标注拟合区域
#     ax1.axvspan(t_min, t_max, color="gray", alpha=0.2, label="Diffusive fit")
#     ax1.legend(loc="upper left")
#     ax2.legend(loc="upper right")

#     plt.title(f"MSD Analysis | D = {D:.3e} Å²/ps")
#     plt.xlim(0, t_max)
#     plt.tight_layout()
#     plt.savefig(filename)
#     plt.show()


# filename = "tmp.msd2"
# time, msd = load_msd(filename)
# mask = time > 100
# slope = compute_local_slope(time[mask], msd[mask])
# D, _, _ = fit_diffusive_region(time[mask], msd[mask], t_min=200, t_max=2500)
# plot_msd_analysis(time[mask], msd[mask], slope, D, t_min=200, t_max=2500, filename="msd_220_tmp")
# # print(time)

import matplotlib.pyplot as plt
import numpy as np


def msd_multi_origin_fft(X):
    # X: (T, N, 3) array of positions for N particles in 3D over T time frames
    T, N, _ = X.shape
    X = X - X.mean(axis=1, keepdims=True)  # Center positions
    msd = np.zeros(T, dtype=np.float64)
    for d in range(3):
        x = X[..., d]
        x2 = (x**2).sum(axis=1)
        # <x(t)x(t+tau)>_t, using FFT
        f = np.fft.rfft(x, n=2 * T, axis=0)
        ac = np.fft.irfft((f * np.conj(f)).sum(axis=1), n=2 * T)[:T] / (np.arange(T, 0, -1))
        msd += 2 * (x2.mean() - ac / N)
    return msd


def load_dump_to_array(filename):
    frames = []
    ids_ref = None
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
                f.readline()
            header = f.readline().strip()  # ITEM: ATOMS id xu yu zu
            cols = header.split()[2:]
            assert cols[:4] == ["id", "xu", "yu", "zu"]
            data = np.loadtxt([f.readline() for _ in range(n)])
            ids = data[:, 0].astype(int)
            xyz = data[:, 1:4]
            if ids_ref is None:
                order = np.argsort(ids)
                ids_ref = ids[order]
            else:
                order = np.argsort(ids)
                assert np.all(ids[order] == ids_ref)
            frames.append(xyz[order])
    X = np.stack(frames, axis=0)  # (T,N,3)
    return X


X = load_dump_to_array("dump_O.lammpstrj")
msd = msd_multi_origin_fft(X)
dt_ps = 1  # dump time interval in ps
t = np.arange(len(msd)) * dt_ps

# # np.savetxt("msd_fft.dat", np.column_stack((t, msd)), header="time(ps) msd(Ang^2)")
# t, msd = np.loadtxt("msd_fft.dat", unpack=True, comments="#")

# # 拟合扩散区（可按局部斜率筛选 τ 区间）
# from scipy.stats import linregress

# mask = (t > 500) & (t < 2000)  # 示例：0.5–2 ns
# slope, intercept, *_ = linregress(t[mask], msd[mask])
# D = slope / 6.0  # Å^2/ps

plt.plot(t[1:], msd[1:], label="MSD", color="blue")
# plt.plot(t[mask], slope * t[mask] + intercept, "r-", lw=2, label=f"fit, D={D:.3e} Å²/ps")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Time (ps)")
plt.ylabel("MSD (Å²)", color="blue")
plt.legend()
plt.tight_layout()
plt.show()
