#!/bin/bash
#SBATCH --job-name=dpmd          # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=4G         # memory per cpu-core (4G per cpu-core is default)
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --time=10:00:00          # total run time limit (HH:MM:SS)


sbatch --dependency=afterok:$SLURM_JOB_ID run.sh

srun singularity exec --nv /home/mcmuniz/Packages/dpmd-jon_test/deepmd-kit_1.3.3_cuda9.2_gpu.sif \
lmp -in lammps_init.in
