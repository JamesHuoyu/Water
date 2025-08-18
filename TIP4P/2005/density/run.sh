# # =============================================
# NP=8                             # MPI进程数
# LAMMPS_SCRIPT="analyze_thermo.lammps"        # LAMMPS输入脚本
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


echo "========================================"
echo "阶段 2: 计算热力学量并可视化"
echo "========================================"

echo "运行命令: python3 analyze_thermo.py"
python3 analyze_thermo.py
# 检查是否成功
if [ $? -ne 0 ]; then
    echo "错误: 热力学量计算失败!"
    exit 1
fi

echo "热力学量计算成功完成!"