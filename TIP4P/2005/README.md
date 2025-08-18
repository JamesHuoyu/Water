in.input 控制粒子初始热浴下的弛豫过程和径向分布函数计算
1H2O 为粒子的构型、拓扑等信息

nbynbyn_H2O.dat 为in.input的输出文件
isf.lammps 用于记录所需条件下的粒子轨迹

tmp.rdf1 为径向分布函数的计算结果
trajectory.lammpstrj 为记录的粒子轨迹

DataArange.py 转换轨迹文本信息至numpy数据结构中

O.npy 为得到的numpy数据结构

structure.py 计算g(r),S(q),isf(q,t)，并保存文件结果至output中