for gamma in 1e-4 5e-4
do
    mpirun -np 8 lmp_tip4p -in /home/debian/water/TIP4P/Ice/shear.lmp -var gamma ${gamma} -var T 225.0 > log.shear_${gamma}_225.lammps
done