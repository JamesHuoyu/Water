import pandas as pd
import os
from tqdm import tqdm


def add_zeta_to_dump(input_file, output_file, zeta_data):
    """
    直接修改 LAMMPS dump 文件，添加 zeta 属性

    参数:
        input_file: 原始 LAMMPS dump 文件路径
        output_file: 输出文件路径
        zeta_data: 包含 zeta 值的 DataFrame，包含 'frame' 和 'O_idx' 列
    """

    # 按帧分组 zeta 数据
    zeta_dict = {}
    for frame, group in zeta_data.groupby("frame"):
        frame = 500000 + frame * 500
        zeta_dict[frame] = dict(zip(group["O_idx"], group["distance"]))

    # 打开输入和输出文件
    with open(input_file, "r") as f_in, open(output_file, "w") as f_out:
        frame_count = -1
        n_atoms = 0
        current_frame = None
        box_lines = []
        header_line = ""
        atom_lines = []
        reading_atoms = False

        # 使用进度条
        total_lines = sum(1 for _ in open(input_file))
        pbar = tqdm(total=total_lines, desc="Processing dump file")

        for line in f_in:
            pbar.update(1)
            line = line.strip()

            if line.startswith("ITEM: TIMESTEP"):
                # 保存前一帧的数据
                if frame_count >= 0:
                    write_frame(
                        f_out, frame_count, n_atoms, box_lines, header_line, atom_lines, zeta_dict
                    )

                # 开始新帧
                frame_count = int(f_in.readline().strip())
                pbar.update(1)
                current_frame = frame_count
                n_atoms = 0
                box_lines = []
                header_line = ""
                atom_lines = []
                reading_atoms = False
                f_out.write(f"ITEM: TIMESTEP\n{frame_count}\n")
                continue

            elif line.startswith("ITEM: NUMBER OF ATOMS"):
                n_atoms_line = f_in.readline().strip()
                pbar.update(1)
                n_atoms = int(n_atoms_line)
                f_out.write(f"ITEM: NUMBER OF ATOMS\n{n_atoms}\n")
                continue

            elif line.startswith("ITEM: BOX BOUNDS"):
                # 读取三行盒子信息
                box_lines.append(line + "\n")
                for _ in range(3):
                    box_line = f_in.readline().strip()
                    pbar.update(1)
                    box_lines.append(box_line + "\n")
                continue

            elif line.startswith("ITEM: ATOMS"):
                header_line = line
                reading_atoms = True
                # 添加 zeta 列到头部
                new_header = header_line + " zeta\n"
                f_out.write(new_header)
                continue

            elif reading_atoms:
                # 收集原子行
                atom_lines.append(line)
                if len(atom_lines) == n_atoms:
                    # 处理完整的一帧
                    write_frame(
                        f_out, current_frame, n_atoms, box_lines, header_line, atom_lines, zeta_dict
                    )
                    reading_atoms = False
                continue

            # 写入其他行
            f_out.write(line + "\n")

        # 处理最后一帧
        if frame_count >= 0 and atom_lines:
            write_frame(f_out, frame_count, n_atoms, box_lines, header_line, atom_lines, zeta_dict)

        pbar.close()


def write_frame(f_out, frame, n_atoms, box_lines, header_line, atom_lines, zeta_dict):
    """写入一帧数据"""
    # 写入盒子信息
    f_out.writelines(box_lines)

    # 获取当前帧的 zeta 值
    frame_zeta = zeta_dict.get(frame, {})

    # 处理每个原子行
    for line in atom_lines:
        parts = line.split()
        atom_id = int(parts[0])

        # 添加 zeta 值（默认为0）
        zeta_value = frame_zeta.get(atom_id, 0.0)

        # 写入新行
        new_line = line + f" {zeta_value}\n"
        f_out.write(new_line)


# 示例使用
if __name__ == "__main__":
    # 原始轨迹文件
    input_dump = "/home/debian/water/TIP4P/2005/2020/4096/traj_2.5e-5_246.lammpstrj"

    # 输出文件
    output_dump = "classified_2.5e-5_246_with_zeta.lammpstrj"

    # 加载 zeta 数据
    zeta_data = pd.read_csv("/home/debian/water/TIP4P/2005/2020/rst/4096/2.5e-5/zeta.csv")

    # 添加 zeta 属性
    add_zeta_to_dump(input_dump, output_dump, zeta_data)

    print(f"处理完成！输出文件: {output_dump}")
