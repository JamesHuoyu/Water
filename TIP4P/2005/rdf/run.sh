#!/bin/bash
# =============================================
# 自动化水分子ISF计算流程脚本
# 功能：执行TIP4P水模型的分子动力学模拟 → SSF计算 → 结果可视化
# 版本：1.0
# 日期：2025-08-20
# =============================================
NP=8                             # MPI进程数
LAMMPS_SCRIPT="in.input"        # LAMMPS输入脚本
# 1. 运行LAMMPS模拟
echo "========================================"
echo "阶段 1: 分子动力学模拟"
echo "========================================"
# 运行LAMMPS
echo "运行命令: mpirun -np ${NP} lmp -in ${LAMMPS_SCRIPT}"
mpirun -np ${NP} lmp -in ${LAMMPS_SCRIPT}

# 检查是否成功
if [ $? -ne 0 ]; then
    echo "错误: LAMMPS模拟失败!"
    exit 1
fi

echo "LAMMPS模拟成功完成!"

echo "激活虚拟环境 myenv"
source ~/myenv/bin/activate

# 2. 计算静态结构因子(SSF)
echo ""
echo "========================================"
echo "阶段 2: 计算静态结构因子(SSF)"
echo "========================================"
# SSF计算参数
RHO=0.03326            # 数密度(Å⁻¹)
K_MAX=25        # k最大值(Å⁻¹)
NK=300             # k值数量
OUTPUT_RDF="tmp.rdf1"  # 输出RDF文件


echo "运行命令: python3 SSF.py ${OUTPUT_RDF} --rho ${RHO} \\"
echo "  --kmax ${K_MAX} --nk ${NK} --window"

python3 SSF.py ${OUTPUT_RDF} --rho ${RHO} --kmax ${K_MAX} --nk ${NK} --window
# 检查是否成功
if [ $? -ne 0 ]; then
    echo "错误: SSF计算失败!"
    exit 1
fi

echo ""
echo "停用虚拟环境"
deactivate
echo "========================================"
echo "流程执行完成!"