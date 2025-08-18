#!/bin/bash
#SBATCH --job-name=dpmd          # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=4G         # memory per cpu-core (4G per cpu-core is default)
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --time=8:00:00          # total run time limit (HH:MM:SS)


# Submit next job and run   
if ! grep -q 'ERROR' slurm* && [ -f "Sampledone.txt" ]; then
    echo "Simulation finished"
elif ! grep -q 'ERROR' slurm*; then
    grep -A 1 "Step TotEng PotEng" log.lammps > data
    value=$(awk 'NR>1 {print$1}' data)
    #echo $value
    mv model_devi.out model_devi.out_$value
    mv density_prof.txt density_prof_$value.txt
    mv log.lammps log.lammps_$value
    mv lmp.lammpstrj lmp.lammpstrj_$value
    rm data
    echo "Continuing running"
    sbatch --dependency=afterok:$SLURM_JOB_ID run.sh
    srun singularity exec --nv /home/mcmuniz/Packages/dpmd-jon_test/deepmd-kit_1.3.3_cuda9.2_gpu.sif \
    lmp -in lammps.in
else
    echo "There is an error"
fi


