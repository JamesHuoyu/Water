# mpirun -np 8 lmp_tip4p -in shear.lmp -var T 225 -var gamma 1e-6 -var Ttotal 50000 > log_1e-6_225.lammps
# mpirun -np 8 lmp_tip4p -in shear.lmp -var T 225 -var gamma 5e-6 -var Ttotal 100000 > log_5e-6_225.lammps
# mpirun -np 8 lmp_tip4p -in shear.lmp -var T 225 -var gamma 1e-5 -var Ttotal 100000 > log_1e-5_225.lammps
# mpirun -np 8 lmp_tip4p -in shear.lmp -var T 225 -var gamma 5e-5 -var Ttotal 100000 > log_5e-5_225.lammps
# mpirun -np 8 lmp_tip4p -in shear.lmp -var T 225 -var gamma 1e-4 -var Ttotal 100000 -var Tpre 20000 > log_1e-4_225.lammps
# mpirun -np 8 lmp_tip4p -in shear.lmp -var T 225 -var gamma 3e-4 -var Ttotal 100000 -var Tpre 10000 > log_3e-4_225.lammps
# mpirun -np 8 lmp_tip4p -in shear.lmp -var T 225 -var gamma 5e-5 -var Tpre 20000 > log.test.lammps
for SEED in 11223 22334 33445 44556 55667 66778 77889 88990 99001 10112 29382 12984 38475 84756 47583 75834 58392 83920 39284 92847 12345 23456 34567 45678 56789 67890 78901 89012 99013 30284
do
  mpirun -np 8 lmp_tip4p -in iso_shear.lmp -var T 225 -var gamma 5e-6 -var seed $SEED -var Ttotal 20000 >> log.iso_shear_250000.lammps
done