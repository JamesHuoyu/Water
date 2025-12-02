import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.colors import LinearSegmentedColormap
from scipy import ndimage
import MDAnalysis as mda
from tqdm import tqdm


def plot_enhanced_heatmap(
    zeta_cg,
    coords,
    sigma=0.7,
    grid_size=150,
    peak_center=0.05,
    peak_width=0.02,
    enhance_contrast=True,
):
    """
    Enhanced heatmap with improved visualization for shoulder peaks around 0.05

    Parameters:
    zeta_cg : np.ndarray: (n_frames*n_points, )
    coords : np.ndarray: (n_frames*n_points, 2)
    sigma : float, Gaussian smoothing parameter
    grid_size : int, resolution of the heatmap grid
    peak_center : float, center of the peak region to enhance
    peak_width : float, width of the peak region to enhance
    enhance_contrast : bool, whether to enhance contrast around peaks
    """
    x = coords[:, 0]
    y = coords[:, 1]
    zeta_values = zeta_cg.flatten()

    # Statistical analysis
    vmin = np.percentile(zeta_values, 2)  # Use 2nd percentile to avoid outliers
    vmax = np.percentile(zeta_values, 98)  # Use 98th percentile
    print(f"zeta_cg statistics: min={np.min(zeta_values):.4f}, max={np.max(zeta_values):.4f}")
    print(f"Using color range: {vmin:.4f} to {vmax:.4f}")
    print(f"Peak region: {peak_center-peak_width/2:.4f} to {peak_center+peak_width/2:.4f}")

    # Create custom colormap that emphasizes the peak region
    if enhance_contrast:
        # Create a colormap with enhanced contrast around the peaks
        cmap = create_enhanced_colormap(peak_center, peak_width, vmin, vmax)
    else:
        cmap = "viridis"

    plt.figure(figsize=(12, 10))

    # Create 2D histogram with weights
    heatmap, xedges, yedges = np.histogram2d(
        x,
        y,
        bins=grid_size,
        weights=zeta_values,
        range=[[np.min(x), np.max(x)], [np.min(y), np.max(y)]],
    )
    counts, _, _ = np.histogram2d(x, y, bins=grid_size)

    # Calculate weighted average
    heatmap = np.divide(heatmap, counts, out=np.zeros_like(heatmap), where=counts != 0)

    # Apply Gaussian smoothing for better visualization
    if sigma > 0:
        heatmap = ndimage.gaussian_filter(heatmap, sigma=sigma)

    # Create the plot
    im = plt.imshow(
        heatmap.T,
        origin="lower",
        aspect="auto",
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        interpolation="bilinear",
    )

    # Add contours to highlight the peak regions
    if enhance_contrast:
        # Add contour lines specifically around the peak region
        peak_min = peak_center - peak_width / 2
        peak_max = peak_center + peak_width / 2

        # Create contour levels that are denser around the peak region
        contour_levels = np.concatenate(
            [
                np.linspace(vmin, peak_min, 5),
                np.linspace(peak_min, peak_max, 8)[1:-1],  # More levels in peak region
                np.linspace(peak_max, vmax, 5),
            ]
        )

        # Remove duplicates and sort
        contour_levels = np.unique(contour_levels)

        # Plot contours
        X, Y = np.meshgrid(
            np.linspace(xedges[0], xedges[-1], grid_size),
            np.linspace(yedges[0], yedges[-1], grid_size),
        )
        contours = plt.contour(
            X, Y, heatmap.T, levels=contour_levels, colors="white", alpha=0.3, linewidths=0.5
        )
        plt.clabel(contours, inline=True, fontsize=8, fmt="%.3f")

    plt.colorbar(im, label="Zeta Potential", shrink=0.8)
    plt.xlabel("X Coordinate (Å)")
    plt.ylabel("Y Coordinate (Å)")

    if enhance_contrast:
        plt.title(
            f"Heatmap of Zeta Potential (Enhanced contrast around {peak_center}±{peak_width/2})"
        )
    else:
        plt.title("Heatmap of Zeta Potential")

    # Add histogram of zeta values to show the distribution
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    n, bins, patches = plt.hist(
        zeta_values, bins=100, alpha=0.7, color="skyblue", edgecolor="black"
    )
    plt.axvline(
        peak_center - peak_width / 2, color="red", linestyle="--", alpha=0.7, label="Peak region"
    )
    plt.axvline(peak_center + peak_width / 2, color="red", linestyle="--", alpha=0.7)
    plt.yscale("log")
    plt.xlabel("Zeta Potential")
    plt.ylabel("Frequency")
    plt.title("Distribution of Zeta Values")
    plt.legend()
    plt.grid(alpha=0.3)

    # Zoom in on the peak region
    plt.subplot(1, 2, 2)
    mask = (zeta_values >= peak_center - peak_width) & (zeta_values <= peak_center + peak_width)
    plt.hist(zeta_values[mask], bins=50, alpha=0.7, color="lightcoral", edgecolor="black")
    plt.axvline(peak_center - peak_width / 2, color="red", linestyle="--", alpha=0.7)
    plt.axvline(peak_center + peak_width / 2, color="red", linestyle="--", alpha=0.7)
    # plt.yscale("log")
    plt.xlabel("Zeta Potential")
    plt.ylabel("Frequency")
    plt.title(f"Zoomed: {peak_center-peak_width:.3f} to {peak_center+peak_width:.3f}")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    # plt.savefig(
    #     "/home/debian/water/TIP4P/2005/2020/rst/equili/zeta_cg_enhanced_heatmap.png", dpi=300
    # )
    plt.show()


def create_enhanced_colormap(peak_center, peak_width, vmin, vmax):
    """
    Create a custom colormap that enhances contrast around specific values
    """
    # Normalize the peak region to [0,1] range
    peak_min_norm = (peak_center - peak_width / 2 - vmin) / (vmax - vmin)
    peak_max_norm = (peak_center + peak_width / 2 - vmin) / (vmax - vmin)
    peak_center_norm = (peak_center - vmin) / (vmax - vmin)

    # Create color segments with enhanced contrast around the peak
    colors_list = [
        (0.0, "darkblue"),  # Low values
        (peak_min_norm * 0.7, "blue"),
        (peak_min_norm, "cyan"),  # Start of peak region
        (peak_center_norm - 0.1, "lightgreen"),
        (peak_center_norm, "yellow"),  # Center of peak - maximum contrast
        (peak_center_norm + 0.1, "orange"),
        (peak_max_norm, "red"),  # End of peak region
        (peak_max_norm + (1 - peak_max_norm) * 0.3, "darkred"),
        (1.0, "maroon"),  # High values
    ]

    # Ensure all values are within [0,1]
    colors_list = [(max(0, min(1, pos)), color) for pos, color in colors_list]

    return LinearSegmentedColormap.from_list("enhanced_cmap", colors_list)


def plot_multiple_views(zeta_cg, coords):
    """
    Plot multiple views with different contrast enhancements
    """
    x = coords[:, 0]
    y = coords[:, 1]
    zeta_values = zeta_cg.flatten()

    vmin = np.percentile(zeta_values, 2)
    vmax = np.percentile(zeta_values, 98)

    # Create different visualization strategies
    strategies = [
        ("Default Viridis", "viridis", False, 0.5),
        ("Enhanced Contrast 0.05", None, True, 0.7),
        ("Hot Colormap", "hot", False, 0.7),
        ("Coolwarm Diverging", "coolwarm", False, 0.7),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for idx, (title, cmap_name, enhance, sigma) in enumerate(strategies):
        ax = axes[idx]

        heatmap, xedges, yedges = np.histogram2d(
            x,
            y,
            bins=100,
            weights=zeta_values,
            range=[[np.min(x), np.max(x)], [np.min(y), np.max(y)]],
        )
        counts, _, _ = np.histogram2d(x, y, bins=100)
        heatmap = np.divide(heatmap, counts, out=np.zeros_like(heatmap), where=counts != 0)

        if sigma > 0:
            heatmap = ndimage.gaussian_filter(heatmap, sigma=sigma)

        if enhance:
            cmap = create_enhanced_colormap(0.05, 0.02, vmin, vmax)
        else:
            cmap = cmap_name

        im = ax.imshow(
            heatmap.T,
            origin="lower",
            aspect="auto",
            extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
        )

        ax.set_title(title)
        ax.set_xlabel("X Coordinate (Å)")
        ax.set_ylabel("Y Coordinate (Å)")
        plt.colorbar(im, ax=ax, shrink=0.8)

    plt.tight_layout()
    plt.savefig("/home/debian/water/TIP4P/2005/nvt/rst/equili/zeta_cg_multiple_views.png", dpi=300)
    plt.show()


# 使用改进的函数 0.45
if __name__ == "__main__":
    # 你的数据加载代码
    zeta_cg = pd.read_csv(
        "/home/debian/water/TIP4P/2005/nvt/rst/equili/zeta_cg.csv", header=None, skiprows=1
    ).to_numpy()
    # print(f"Headers of zeta_cg data: {zeta_cg[:5,:5]}")  # 打印前5行5列以检查数据格式
    # print(f"dtype of zeta_cg: {zeta_cg.dtype}, shape: {zeta_cg.shape}")
    file_path = "/home/debian/water/TIP4P/2005/nvt/dump_225_test.lammpstrj"
    u = mda.Universe(file_path, format="LAMMPSDUMP")
    from MDAnalysis.transformations import wrap

    u.trajectory.add_transformations(wrap(u.atoms))

    O_atoms = u.select_atoms("type 1")
    # coords = np.zeros((501, len(O_atoms), 3))
    len_coords = len(u.trajectory)
    coords = np.zeros((len_coords, len(O_atoms), 3))
    # frame_start = 9500
    frame_start = 0
    # 只加载观察一个t_x时间内的heatmap数据
    for ts in u.trajectory[frame_start:]:
        coords[ts.frame - frame_start] = O_atoms.positions

    coords_reshaped = coords.reshape((-1, 3))
    print("Plotting enhanced heatmap...")
    t_x = 38.00  # ps
    time_step = 0.2  # ps
    frame_x = int(t_x / time_step)

    idx = frame_x * len(O_atoms)
    print(f"Using data up to frame {frame_x} (index {idx}) for heatmap.")
    print(f"zeta_cg shape: {zeta_cg.shape}, coords shape: {coords_reshaped.shape}")
    # 使用增强对比度的版本
    plot_enhanced_heatmap(
        zeta_cg[:idx, -1],
        coords_reshaped[:idx, :2],
        peak_center=0.05,
        peak_width=0.02,
        enhance_contrast=True,
    )

    print("Plotting multiple views for comparison...")
    # 比较不同可视化方法
    plot_multiple_views(zeta_cg[:idx, -1], coords_reshaped[:idx, :2])
