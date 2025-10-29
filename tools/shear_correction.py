import numpy as np
import MDAnalysis as mda
from MDAnalysis.lib.distances import apply_PBC
from tqdm import tqdm


def remove_affine_deformation(positions, box0, box1):
    affine_matrix = np.linalg.inv(box0) @ box1
    return positions @ affine_matrix.T


def write_lammps_dump_frame(f, step, box, positions, atom_ids, atom_types):
    """写入LAMMPS轨迹帧"""
    f.write(f"ITEM: TIMESTEP\n{step}\n")
    f.write(f"ITEM: NUMBER OF ATOMS\n{len(positions)}\n")
    f.write("ITEM: BOX BOUNDS pp pp pp\n")
    # 写入正交盒子信息
    f.write(f"{box[0][0]:.15e} {box[0][1]:.15e}\n")
    f.write(f"{box[1][0]:.15e} {box[1][1]:.15e}\n")
    f.write(f"{box[2][0]:.15e} {box[2][1]:.15e}\n")
    f.write("ITEM: ATOMS id type xu yu zu\n")
    # 写入原子信息
    for i in range(len(positions)):
        atom_id = atom_ids[i]
        atom_type = atom_types[i]
        x, y, z = positions[i]
        f.write(f"{atom_id} {atom_type} {x:.6g} {y:.6g} {z:.6g}\n")


def apply_shear_flow_correction(dump_file, strain_rate, time_step, frame_step, output_file):
    """
    应用剪切流校正，通过反剪切变换和unwrapping去除对流影响
    :param dump_file: 输入轨迹文件
    :param strain_rate: 剪切速率 (fs⁻¹)
    :param time_step: 时间步长 (fs)
    :param frame_step: 帧间隔 (步数)
    :param output_file: 输出文件
    """
    u = mda.Universe(dump_file, format="LAMMPSDUMP")

    # 获取原子信息
    atom_ids = u.atoms.ids
    atom_types = u.atoms.types

    # 初始化穿越次数记录
    images = np.zeros((len(u.atoms), 3), dtype=np.int32)

    # 获取初始帧信息
    initial_frame = u.trajectory[0]
    initial_step_value = initial_frame.frame

    # 打开输出文件
    with open(output_file, "w") as f:
        for ts in tqdm(u.trajectory[0:2], desc="Correcting shear flow"):
            # 计算当前时间
            current_step = ts.frame
            delta_time = (current_step - initial_step_value) * frame_step * time_step

            # 计算总应变
            strain = strain_rate * delta_time

            # 应用PBC折叠原子到盒子内 (wrapping)
            wrapped_positions = apply_PBC(ts.positions, ts.dimensions)

            # 更新穿越次数
            # MDAnalysis没有直接提供更新image flags的函数，需要手动计算
            # 这里简化处理：实际应用中需要更精确的穿越次数跟踪
            displacement = ts.positions - wrapped_positions
            print(ts.dimensions)
            print(ts.triclinic_dimensions)
            box_dims = np.diag(ts.triclinic_dimensions)
            print(displacement, box_dims)
            images += np.floor(displacement / box_dims + 0.5).astype(np.int32)

            # 应用反剪切变换到折叠位置和盒子
            unsheared_pos, unsheared_box = apply_unshear_transformation(
                wrapped_positions, ts.triclinic_dimensions, strain
            )

            # 在反剪切后的正交盒子中应用unwrapping
            unwrapped_positions = unwrap(
                unsheared_pos,
                unsheared_box,
                images=images,
                # 使用正交盒子，所以设置pbc=(True, True, True)
                # 盒子尺寸为 unsheared_box 的对角线元素
                box=unsheared_box.diagonal(),
                pbc=(True, True, True),
            )

            # 写入当前帧
            write_lammps_dump_frame(
                f, current_step, unsheared_box, unwrapped_positions, atom_ids, atom_types
            )


if __name__ == "__main__":
    dump_file = "/home/debian/water/TIP4P/2005/2020/4096/traj_2.5e-5_246.lammpstrj"
    output_file = (
        "/home/debian/water/TIP4P/2005/2020/4096/traj_2.5e-5_246_shear_corrected.lammpstrj"
    )
    strain_rate = 2.5e-5  # fs⁻¹
    time_step = 1.0  # fs
    frame_step = 500  # 每隔500帧保存一次
    apply_shear_flow_correction(dump_file, strain_rate, time_step, frame_step, output_file)
    print(f"Shear correction applied. Corrected trajectory saved to {output_file}.")
