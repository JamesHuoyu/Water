import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd


def plot_thermo_vs_t(thermos, timestep, bias=0):
    """
    Plots thermo vs temperature for a given set of densities.

    Parameters:
    densities (list or np.ndarray): List or array of density values.
    """
    plt.figure(figsize=(8, 6))
    plt.plot((thermos[:, 0] - bias) * timestep, thermos[:, 1], marker="o", linestyle="", color="b")
    plt.title("Density vs time")
    plt.xlabel("time")
    plt.ylabel("Density (g/cm³)")
    plt.grid(True)
    plt.show()


def create_dataframe(block_data):
    columns = ["row", "r", "rdf_OO", "n_OO"]
    df = pd.DataFrame(block_data, columns=columns)
    df.drop(columns=["row"], inplace=True)

    df.set_index("r", inplace=True)

    return df


def parse_rdf_vector_file(fname, lenperblock=8, last_n_frames=None):
    """Read tmp.rdf (mode vector). Returns r (Å), dict of g_ab, dict of coord_ab."""
    data_blocks = []
    current_block = []
    nrows = None
    with open(fname, "r") as f:
        for line in f:
            if not line or line.startswith("#"):
                continue

            parts = line.split()

            if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                if current_block:
                    data_blocks.append((nrows, current_block))
                    current_block = []

                nrows = int(parts[1])

            elif len(parts) == lenperblock and parts[0].isdigit():
                row_data = list(map(float, parts))
                current_block.append(row_data)

    if current_block:
        data_blocks.append((nrows, current_block))

    if last_n_frames is not None:
        data_blocks = data_blocks[-last_n_frames:]

    all_data = []
    for nrow, block in data_blocks:
        cur_df = create_dataframe(block)
        if nrow != nrows:
            raise ValueError("Cannot do average rdf between different time periods")
        else:
            all_data.append(cur_df)
    grouped = pd.concat(all_data).groupby(level=0)
    mean_df = grouped.mean()
    std_df = grouped.std()

    mean_df.columns = [col + "_mean" for col in mean_df.columns]
    std_df.columns = [col + "_std" for col in std_df.columns]

    final_df = pd.concat([mean_df, std_df], axis=1)

    return final_df


def plot_with_error(x, y, yerr, ylabel):
    plt.errorbar(x, y, yerr=yerr, fmt="o-", capsize=3)
    plt.xlabel(r"$k$ ($\mathrm{\AA}^{-1}$)")
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    # plt.savefig(
    # f"{ylabel.replace(' ', '_').replace('$','').replace('{','').replace('}','').lower()}.png",
    # dpi=300,
    # )
    plt.show()
    # plt.close()


thermos = np.loadtxt("./results/db_rs/tmp.rho")
timestep_fs = 2
timestep_ps = timestep_fs / 1000.0
timestep_ns = timestep_ps / 1000.0
bias = 3e7
block_size = 250
num_blocks = thermos.shape[0] // block_size
means = []
variances = []
for i in range(num_blocks):
    start_idx = i * block_size
    end_idx = start_idx + block_size

    block = thermos[start_idx:end_idx, 1]
    means.append(np.mean(block))
    variances.append(np.var(block, ddof=1))
means = np.array(means)
variances = np.array(variances)

print(means)
print(variances)
plot_with_error(
    np.arange(num_blocks) * block_size * timestep_ps,
    means,
    np.sqrt(variances / block_size),
    ylabel="Density (g/cm³)",
)
# plot_thermo_vs_t(thermos, timestep_ns, bias)
# temp = np.loadtxt("tmp.temp")
# plot_thermo_vs_t(temp, timestep_ps)
# rdf = parse_rdf_vector_file("./results/tmp.rdf1", lenperblock=4)
# plot_with_error(rdf.index.values, rdf["rdf_OO_mean"], rdf["rdf_OO_std"], ylabel=r"$g_{OO}(r)$")
