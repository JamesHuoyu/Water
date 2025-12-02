import numpy as np
import matplotlib.pyplot as plt
import MDAnalysis as mda
import pandas as pd
from tqdm import tqdm
from numba import jit, prange
import argparse
import os
import h5py
from abc import ABC, abstractmethod
from typing import Tuple, Optional, Dict, Any, List, Union


class MDEnsembleBase(ABC):
    def __init__(
        self,
        universe: mda.Universe,
        atom_selection: str = "type 1",
        shear_rate: float = 0.0,
        time_step: float = 1.0,
        start_index: int = 0,
        unwrap: bool = True,
        **kwargs,
    ):
        self.universe = universe
        self.atom_selection = atom_selection
        self.shear_rate = shear_rate
        self.time_step = time_step
        self.start_index = start_index
        self.unwrap = unwrap
        self._init_subclass_params(**kwargs)
        self.target_atoms = self.universe.select_atoms(self.atom_selection)
        self.n_atoms = len(self.target_atoms)

        self._load_trajectory()
        if self.shear_rate != 0.0:
            self._apply_shear_correction()

        self._subclass_init()

    def _init_subclass_params(self, **kwargs):
        self._kwargs = kwargs

    def _subclass_init(self):
        pass

    def _load_trajectory(self):
        self.n_frames = len(self.universe.trajectory)
        self.frames = self.n_frames - self.start_index

        if self.frames <= 0:
            raise ValueError("No frames to process after applying start_index.")

        self.coords = np.zeros((self.frames, self.n_atoms, 3))
        self.box_dims = np.zeros((self.frames, 6))

        print(f"Loading trajectory with {self.frames} frames... with unwrapping: {self.unwrap}")

        for i, ts in enumerate(
            tqdm(self.universe.trajectory[self.start_index :], desc="Loading trajectory")
        ):
            self.coords[i] = self.target_atoms.positions
            self.box_dims[i] = ts.dimensions

    def _apply_shear_correction(self):
        if not self.unwrap:
            print("Warning: Shear correction applied without unwrapping coordinates.")
        print(f"Applying shear correction (rate: {self.shear_rate} 1/ps)...")
        for frame in tqdm(range(self.frames), desc="Applying shear correction"):
            y_pos = self.coords[frame - 1, :, 1]
            self.coords[frame:, :, 0] -= self.shear_rate * y_pos * self.time_step

    @abstractmethod
    def compute_single_origin(self, t0: int, max_tau: int) -> np.ndarray:
        pass

    def time_origin_average(self, max_tau: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        max_tau = max_tau or self.frames

        if max_tau > self.frames:
            raise ValueError("max_tau cannot be greater than the number of frames.")
            max_tau = self.frames

        test_result = self.compute_single_origin(0, min(10, max_tau))

        if test_result.ndim == 1:
            n_quantities = 1
            result_shape = (max_tau,)
        else:
            n_quantities, _ = test_result.shape
            result_shape = (n_quantities, max_tau)

        accum = np.zeros(result_shape)
        count = np.zeros(max_tau, dtype=int)

        max_t0 = self.frames - max_tau

        if max_t0 < 0:
            raise ValueError(
                "Not enough frames to perform time-origin averaging with the given max_tau."
            )

        progress_desc = f"Computing {self.quantity_name}"
        for t0 in tqdm(range(max_t0), desc=progress_desc):
            result = self.compute_single_origin(t0, max_tau)
            valid_tau = min(result.shape[-1], max_tau)
            if n_quantities == 1:
                accum[:valid_tau] += result[:valid_tau]
            else:
                accum[:, :valid_tau] += result[:, :valid_tau]

            count[:valid_tau] += 1
        valid_mask = count > 0
        avg_result = np.zeros(result_shape)

        if n_quantities == 1:
            avg_result[valid_mask] = accum[valid_mask] / count[valid_mask]
        else:
            for i in range(n_quantities):
                avg_result[i, valid_mask] = accum[i, valid_mask] / count[valid_mask]

        times = np.arange(max_tau) * self.time_step

        return avg_result, times

    def save_results(
        self,
        results: np.ndarray,
        times: np.ndarray,
        output_file: str,
        dataset_name: str,
        **metadata,
    ):
        """
        保存结果到HDF5文件（基类实现）

        :param results: 结果数组
        :param times: 时间数组
        :param output_file: 输出文件路径
        :param dataset_name: 数据集名称
        :param metadata: 元数据
        """
        # 确保输出目录存在
        os.makedirs(
            os.path.dirname(output_file) if os.path.dirname(output_file) else ".", exist_ok=True
        )

        with h5py.File(output_file, "a") as f:
            # 创建组
            if dataset_name not in f:
                group = f.create_group(dataset_name)
            else:
                group = f[dataset_name]

            # 保存主数据集
            if results.ndim == 1:
                if "values" in group:
                    del group["values"]
                group.create_dataset("values", data=results)
            else:
                for i in range(results.shape[0]):
                    ds_name = f"values_{i}" if results.shape[0] > 1 else "values"
                    if ds_name in group:
                        del group[ds_name]
                    group.create_dataset(ds_name, data=results[i])

            # 保存时间轴
            if "times" in group:
                del group["times"]
            group.create_dataset("times", data=times)

            # 保存元数据
            for key, value in metadata.items():
                if isinstance(value, (str, int, float, bool)):
                    group.attrs[key] = value

            # 保存分析器基本信息
            group.attrs["quantity_name"] = self.quantity_name
            group.attrs["n_atoms"] = self.n_atoms
            group.attrs["n_frames"] = self.frames
            group.attrs["time_step"] = self.time_step
            group.attrs["shear_rate"] = self.shear_rate
            group.attrs["unwrap_coords"] = self.unwrap_coords

    def plot_results(
        self,
        results: np.ndarray,
        times: np.ndarray,
        output_file: Optional[str] = None,
        **plot_kwargs,
    ):
        """
        绘制结果（基类实现）

        :param results: 结果数组
        :param times: 时间数组
        :param output_file: 输出文件路径（可选）
        :param plot_kwargs: 绘图参数
        """
        plt.figure(figsize=(10, 6))

        if results.ndim == 1:
            plt.plot(times, results, **plot_kwargs)
        else:
            for i in range(results.shape[0]):
                label = plot_kwargs.pop("label", f"Component {i}") if i == 0 else f"Component {i}"
                plt.plot(times, results[i], label=label, **plot_kwargs)
            plt.legend()

        plt.xlabel("Time (ps)")
        plt.ylabel(self.quantity_name)
        plt.title(f"{self.quantity_name} vs Time")
        plt.grid(True, alpha=0.3)

        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches="tight")
            print(f"Plot saved to {output_file}")

        plt.show()

    @property
    @abstractmethod
    def quantity_name(self) -> str:
        pass


class AnalysisRunner:
    pass
