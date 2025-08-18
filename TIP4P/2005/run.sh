#!/bin/bash
# =============================================
# 自动化水分子ISF计算流程脚本
# 功能：执行TIP4P水模型的分子动力学模拟 → ISF计算 → 结果可视化
# 版本：1.0
# 日期：2025-08-13
# =============================================
# NP=8                             # MPI进程数
# LAMMPS_SCRIPT="isf.lammps"        # LAMMPS输入脚本
# # 1. 运行LAMMPS模拟
# echo "========================================"
# echo "阶段 1: 分子动力学模拟"
# echo "========================================"
# # 运行LAMMPS
# echo "运行命令: mpirun -np ${NP} lmp -in ${LAMMPS_SCRIPT}"
# mpirun -np ${NP} lmp -in ${LAMMPS_SCRIPT}

# # 检查是否成功
# if [ $? -ne 0 ]; then
#     echo "错误: LAMMPS模拟失败!"
#     exit 1
# fi

# echo "LAMMPS模拟成功完成!"

echo "激活虚拟环境 myenv"
source ~/myenv/bin/activate

# 2. 计算中间散射函数(ISF)
echo ""
echo "========================================"
echo "阶段 2: 计算中间散射函数(ISF)"
echo "========================================"
# ISF计算参数
DT_FS=10            # 时间步长(fs)
K_TARGET=2.0        # 目标k值(Å⁻¹)
DK=0.15             # k-shell宽度(Å⁻¹)
TMAX_PS=500         # 最大时间延迟(ps)
STRIDE_PS=4       # 原点时间间隔(ps)
NLOG=250
MODE=coherent       # 模式选择
OUTPUT_TRAJ="isf_oxy_280.lammpstrj"  # 输出轨迹文件


echo "运行命令: python3 isf.py --dumps ${OUTPUT_TRAJ} --dt_fs ${DT_FS} \\"
echo "  --k_target ${K_TARGET} --dk ${DK} --tmax_ps ${TMAX_PS} --origin_stride_ps ${STRIDE_PS}"

python3 isf.py --dumps ${OUTPUT_TRAJ} --dt_fs ${DT_FS} \
  --k_target ${K_TARGET} --dk ${DK} --tmax_ps ${TMAX_PS} --origin_stride_ps ${STRIDE_PS} \
  --logtime --n_log ${NLOG} --mode ${MODE}

# 检查是否成功
if [ $? -ne 0 ]; then
    echo "错误: ISF计算失败!"
    exit 1
fi

# 3. 结果可视化
echo ""
echo "========================================"
echo "阶段 3: 结果可视化"
echo "========================================"
python3 visualize.py

# 检查是否成功
if [ $? -ne 0 ]; then
    echo "错误: 可视化失败!"
    exit 1
fi

echo ""
echo "停用虚拟环境"
deactivate
echo "========================================"
echo "流程执行完成!"