import MDAnalysis as mda
from MDAnalysis.analysis.msd import EinsteinMSD
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from scipy.stats import linregress


def calculate_msd(universe, start_time, end_time):
    O_atoms = universe.select_atoms("type 1")
    n_frames = len(universe.trajectory)
    n_water = len(O_atoms)
    positions = np.zeros((n_frames, n_water, 3))

    for i, ts in enumerate(universe.trajectory):
        positions[i] = O_atoms.positions  # 使用 unwrapped 坐标

    msd_values = []
    for t in range(start_time, end_time):
        displacements = positions[t:] - positions[:-t]  # 计算位移
        squared_displacements = np.sum(displacements**2, axis=2)  # 每个原子平方位移
        msd = np.mean(squared_displacements)  # 计算 MSD
        msd_values.append(msd)

    return np.array(msd_values)


if __name__ == "__main__":
    dump_file = "/home/debian/water/TIP4P/2005/benchmark/220/quenching/dump_H2O.lammpstrj"
    print("Loading trajectory...")
    u = mda.Universe(dump_file, format="LAMMPSDUMP")
    print("Running MSD analysis...")
    msd_analysis = EinsteinMSD(
        u,
        select="type 1",
        msd_type="xyz",
        fft=True,
        time_origin="all",
    )
    msd_analysis.run()
    bias = 15500000
    times = (msd_analysis.times - bias) * 0.002  # Convert to ps assuming timestep of 2 fs
    print(times)
    msd_values = msd_analysis.results.timeseries
    plt.figure(figsize=(8, 6))
    plt.plot(times, msd_values, label="MSD")
    plt.xlabel("Time (ps)")
    plt.ylabel("Mean Squared Displacement (Å²)")
    plt.title("Mean Squared Displacement vs Time")
    plt.legend()
    plt.savefig("quenching/msd_plot.png", dpi=300)
    plt.show()
    pd.DataFrame({"time_ps": times, "msd_A2": msd_values}).to_csv(
        "quenching/msd_data.csv", index=False
    )

    df = pd.read_csv("quenching/msd_data.csv")
    times = df["time_ps"].values
    msd_values = df["msd_A2"].values
    print("Computing self-diffusion coefficient...")
    start_time = 20000  # ps
    end_time = 50000  # ps
    linear_model = linregress(
        times[(times >= start_time) & (times <= end_time)],
        msd_values[(times >= start_time) & (times <= end_time)],
    )
    slope = linear_model.slope
    error = linear_model.stderr
    diffusion_coefficient = slope / 6  # D = slope / 6 for 3D diffusion
    diffusion_error = error / 6
    print(f"Self-diffusion coefficient: {diffusion_coefficient:.5e} ± {diffusion_error:.5e} Å²/ps")
    print("Converting to m²/s...")
    print(
        f"Self-diffusion coefficient: {diffusion_coefficient * 1e-8:.5e} ± {diffusion_error * 1e-8:.5e} m²/s"
    )
    print("Plotting MSD data...")
    times_ns = times / 1000  # Convert to ns
    plt.figure(figsize=(8, 6))
    plt.plot(times, msd_values, label="MSD")
    plt.xlabel("Time (ps)")
    plt.ylabel("Mean Squared Displacement (Å²)")
    plt.xscale("log")
    plt.yscale("log")
    plt.axvline(start_time, color="red", linestyle="--", label="Fit Start")
    plt.axvline(end_time, color="green", linestyle="--", label="Fit End")

    # 对时间和MSD取对数
    mask = (times >= start_time) & (times <= end_time)
    log_times = np.log10(times[mask])
    log_msd_values = np.log10(msd_values[mask])

    # 执行线性拟合
    slope, intercept, r_value, p_value, std_err = linregress(log_times, log_msd_values)
    print(f"Log-Log Linear Fit: slope = {slope}, intercept = {intercept}, R² = {r_value**2}")
    plt.plot(
        times[mask],
        10 ** (intercept) * times[mask] ** slope,
        color="orange",
        linestyle="--",
        label=f"Log-Log Fit: slope={slope:.2f}",
    )
    plt.title("Mean Squared Displacement vs Time")
    plt.legend()
    plt.savefig("quenching/msd_plot_loglog.png", dpi=300)
    plt.show()
