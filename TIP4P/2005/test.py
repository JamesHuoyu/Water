import numpy as np
import matplotlib.pyplot as plt
import MDAnalysis as mda
from MDAnalysis.analysis.hydrogenbonds.hbond_analysis import HydrogenBondAnalysis as HBA
from collections import defaultdict

# txt_file = "/home/debian/water/TIP4P/2005/2020/4096/stress_2.5e-5_246.data"
# step, strain, stress = np.loadtxt(txt_file, comments="#", unpack=True)
# strain_rate = (strain[1] - strain[0]) * 1e15 / (step[1] - step[0])  # in fs⁻¹
# viscosity = stress * 1e9 / strain_rate  # in Pa·s
# plt.figure(figsize=(8, 6))
# plt.plot(step, viscosity, "b-", linewidth=1.5)
# plt.xlabel("Time Step")
# plt.ylabel("Viscosity (Pa·s)")
# plt.title(f"Viscosity Evolution at T=246K, Strain Rate: {strain_rate:.2e} fs⁻¹")
# plt.grid(True, linestyle="--", alpha=0.7)
# window_size = max(100, int(len(viscosity) * 0.01))
# # viscosity = np.convolve(viscosity, np.ones(window_size) / window_size, mode="valid")
# start_idx = int(len(viscosity) * 0.3)
# end_idx = int(len(viscosity) * 0.75)
# average_viscosity = np.mean(viscosity[start_idx:end_idx])
# rms_viscosity = np.std(viscosity[start_idx:end_idx])
# plt.axhline(
#     y=average_viscosity, color="r", linestyle="--", label=f"Avg: {average_viscosity:.4f} Pa·s"
# )
# plt.fill_between(
#     step,
#     average_viscosity - rms_viscosity,
#     average_viscosity + rms_viscosity,
#     color="r",
#     alpha=0.2,
# )
# plt.legend(loc="best")
# print(f"{txt_file}: Average viscosity: {average_viscosity:.5f} ± {rms_viscosity:.5f} Pa·s")
# # plt.savefig("viscosity_246K_2.5e-5_4096.png", dpi=300)
# plt.show()

# u = mda.Universe(
#     "/home/debian/water/TIP4P/2005/2020/4096/traj_2.5e-5_246.lammpstrj", format="LAMMPSDUMP"
# )
# O_atoms = u.select_atoms("type 1")
# global_O_indices = O_atoms.indices
# n_frames = len(u.trajectory)
# print(f"Total number of frames: {n_frames}")

# print("Running hydrogen bond analysis...")
# hbond_analysis = HBA(
#     u,
#     donors_sel="type 1",
#     hydrogens_sel="type 2",
#     acceptors_sel="type 1",
#     d_a_cutoff=3.5,
#     d_h_a_angle_cutoff=150.0,
# )
# hbond_analysis.run()
# hbonds = hbond_analysis.results.hbonds
# frame_counts = defaultdict(lambda: defaultdict(int))

# for hbond in hbonds:
#     frame, donor_idx, _, acceptor_idx, _, _ = hbond
#     frame = int(frame)
#     donor_idx = int(donor_idx)
#     acceptor_idx = int(acceptor_idx)
#     frame_counts[frame][donor_idx] += 1
#     frame_counts[frame][acceptor_idx] += 1

# with open("/home/debian/water/TIP4P/2005/2020/rst/4096/hbond_246K_2.5e-5.data", "w") as f:
#     f.write("frame, O_index, n_hbonds\n")
#     for frame in sorted(frame_counts.keys()):
#         for O_idx in sorted(frame_counts[frame].keys()):
#             n_hbonds = frame_counts[frame][O_idx]
#             f.write(f"{frame}, {O_idx}, {n_hbonds}\n")

# print("Hydrogen bond analysis completed.")

# csv_file = "/home/debian/water/TIP4P/2005/2020/rst/4096/hbond_246K_2.5e-5.data"
# data = np.loadtxt(csv_file, delimiter=",", skiprows=1)
# # 绘制氢键数随时间变化的折线图
# average_hbonds_per_frame = []
# plt.figure(figsize=(10, 6))
# for frame in np.unique(data[:, 0]):
#     mask = data[:, 0] == frame
#     average_hbonds = np.mean(data[mask, 2])
#     average_hbonds_per_frame.append(average_hbonds)
# plt.plot(
#     np.unique(data[:, 0]),
#     average_hbonds_per_frame,
#     "b-",
#     linewidth=1.5,
#     label="Average Hydrogen Bonds per O Atom",
# )
# plt.xlabel("Frame")
# plt.ylabel("Number of Hydrogen Bonds")
# plt.title("Hydrogen Bonds per Oxygen Atom Over Time at 246K")
# plt.legend(loc="upper right", fontsize="small", ncol=2)
# plt.grid(True, linestyle="--", alpha=0.7)
# # plt.savefig("hbond_time_series_246K.png", dpi=300)
# plt.show()
# hbond_counts = []
# with open("/home/debian/water/TIP4P/2005/2020/rst/4096/hbond_246K_2.5e-5.data", "r") as f:
#     next(f)  # Skip header line
#     for line in f:
#         _, _, count = line.strip().split(", ")
#         hbond_counts.append(int(count))

# plt.figure(figsize=(8, 6))
# plt.hist(
#     hbond_counts,
#     bins=range(0, max(hbond_counts) + 2),
#     density=True,
#     alpha=0.7,
#     align="left",
#     rwidth=0.8,
# )
# average_hbonds = np.mean(hbond_counts)
# plt.axvline(x=average_hbonds, color="r", linestyle="--", label=f"Avg: {average_hbonds:.2f}")
# plt.legend(loc="best")
# plt.xlabel("Number of Hydrogen Bonds")
# plt.ylabel("Probability Density")
# plt.title("Hydrogen Bond Distribution Over Time")
# plt.grid(True, linestyle="--", alpha=0.7)
# # plt.savefig("hbond_distribution_246K.png", dpi=300)
# plt.show()


txt_list = [
    "/home/debian/water/TIP4P/2005/2020/4096/multi/stress_5e-7_246.data",
    "/home/debian/water/TIP4P/2005/2020/4096/multi/stress_2.5e-6_246.data",
    "/home/debian/water/TIP4P/2005/2020/4096/multi/stress_5e-6_246.data",
    "/home/debian/water/TIP4P/2005/2020/4096/stress_2.5e-5_246.data",
    "/home/debian/water/TIP4P/2005/2020/4096/multi/stress_2.5e-4_246.data",
]
# txt_file = "/home/debian/water/TIP4P/2005/stress_5.0e-5_246.data"
viscoisties = []
strain_rates = []
for txt_file in txt_list:
    step, strain, stress = np.loadtxt(txt_file, comments="#", unpack=True)
    strain_rate = (strain[1] - strain[0]) * 1e15 / (step[1] - step[0])  # in fs⁻¹
    viscosity = stress * 1e9 / strain_rate  # in Pa·s
    plt.figure(figsize=(8, 6))
    plt.plot(step, viscosity, "b-", linewidth=1.5)
    plt.xlabel("Time Step")
    plt.ylabel("Viscosity (Pa·s)")
    plt.title(f"Viscosity Evolution at T=246K, Strain Rate: {strain_rate*1e-15:.2e} fs⁻¹")
    plt.grid(True, linestyle="--", alpha=0.7)
    # half_idx = len(viscosity) // 3
    window_size = max(100, int(len(viscosity) * 0.01))
    viscosity = np.convolve(viscosity, np.ones(window_size) / window_size, mode="valid")
    start_idx = int(len(viscosity) * 0.3)
    average_viscosity = np.mean(viscosity[start_idx:])
    rms_viscosity = np.std(viscosity[start_idx:])
    plt.axhline(
        y=average_viscosity, color="r", linestyle="--", label=f"Avg: {average_viscosity:.4f} Pa·s"
    )
    plt.fill_between(
        step,
        average_viscosity - rms_viscosity,
        average_viscosity + rms_viscosity,
        color="r",
        alpha=0.2,
    )
    plt.legend(loc="best")
    print(f"{txt_file}: Average viscosity: {average_viscosity:.5f} ± {rms_viscosity:.5f} Pa·s")
    # plt.savefig("viscosity_246K_2.5e-5.png", dpi=300)
    plt.show()
    viscoisties.append((average_viscosity, rms_viscosity))
    strain_rates.append(strain_rate)
plt.figure(figsize=(8, 6))
plt.errorbar(
    strain_rates, [v[0] for v in viscoisties], yerr=[v[1] for v in viscoisties], fmt="o-", capsize=5
)
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Strain Rate (fs⁻¹)")
plt.ylabel("Average Viscosity (Pa·s)")
plt.title("Viscosity vs Strain Rate at T=246K")
plt.grid(True, linestyle="--", alpha=0.7)
# plt.savefig("viscosity_vs_strain_rate_246K.png", dpi=300)
plt.show()
