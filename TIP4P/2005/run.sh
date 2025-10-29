export OMP_NUM_THREADS=2
export KMP_AFFINITY=granularity=fine,compact,1,0
export KMP_BLOCKTIME=0
T="246"

# mpirun -np 8 lmp_tip4p -in input.lmp -var T $T -sf intel
# # mpirun -np 8 lmp_tip4p -in input.lmp -var T $T
# mpirun -np 8 lmp_tip4p -in nvt.lmp -var T 246 -var Lx_avg 24.89526 -var Ly_avg 24.89526 -var Lz_avg 24.89526 -sf intel
mpirun -np 8 lmp_tip4p -in shear.lmp -var T 246 -var gamma 2.5e-5 > log.2.5e-5.lammps
mpirun -np 8 lmp_tip4p -in shear.lmp -var T 246 -var gamma 5.0e-5 > log.5e-5.lammps
mpirun -np 8 lmp_tip4p -in shear.lmp -var T 246 -var gamma 2.5e-6 > log.2.5e-6.lammps
mpirun -np 8 lmp_tip4p -in shear.lmp -var T 246 -var gamma 5.0e-6 > log.5.0e-6.lammps
mpirun -np 8 lmp_tip4p -in shear.lmp -var T 246 -var gamma 1.0e-5 > log.1.0e-5.lammps
mpirun -np 8 lmp_tip4p -in shear.lmp -var T 246 -var gamma 1.0e-7 > log.1.0e-7.lammps
mpirun -np 8 lmp_tip4p -in shear.lmp -var T 246 -var gamma 2.0e-7 > log.2.0e-7.lammps
mpirun -np 8 lmp_tip4p -in shear.lmp -var T 246 -var gamma 5.0e-7 > log.5.0e-7.lammps
mpirun -np 8 lmp_tip4p -in shear.lmp -var T 246 -var gamma 1.0e-8 > log.1.0e-8.lammps
mpirun -np 8 lmp_tip4p -in shear.lmp -var T 246 -var gamma 2.0e-8 > log.2.0e-8.lammps
mpirun -np 8 lmp_tip4p -in shear.lmp -var T 246 -var gamma 5.0e-8 > log.5.0e-8.lammps