import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os


def plot_hb_distribution(hb_counts_file):
    print("[5] Plotting distributions ...")
    # hb_counts = []
    # with open(hb_counts_file, "r") as f:
    #     next(f)
    #     for line in f:
    #         _, _, c = line.strip().split(",")
    #         hb_counts.append(int(c))

    # if len(hb_counts) == 0:
    #     print("No hb count data to plot")
    #     return
    df = pd.read_csv(hb_counts_file)

    plt.figure(figsize=(8, 5))
    plt.hist(
        df["hb_count"],
        bins=range(0, df["hb_count"].max() + 2),
        density=True,
        align="left",
        rwidth=0.8,
    )
    plt.xlabel("Number of Hydrogen Bonds per Oxygen")
    plt.ylabel("Probability Density")
    plt.title("Distribution of Hydrogen Bonds per Oxygen")
    plt.grid(alpha=0.3)
    plt.text(
        0.95,
        0.95,
        f"Average: {np.mean(df['hb_count']):.2f}",
        transform=plt.gca().transAxes,
        ha="right",
        va="top",
    )
    plt.savefig("hb_count_distribution.png", dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    hb_counts_file = "/home/debian/water/TIP4P/Ice/225/shear/rst/5e-4/hb_counts_per_idx.csv"
    if os.path.exists(hb_counts_file):
        plot_hb_distribution(hb_counts_file)
    else:
        print(f"File {hb_counts_file} does not exist.")
