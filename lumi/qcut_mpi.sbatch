#!/bin/bash
# QCut MPI task-farm on LUMI-C (CPU partition).
#
# Regime A: many ranks, 1 thread each. Best for circuit-cutting ensembles, which
# are thousands of *small* circuits — MPI provides the parallelism, Aer runs each
# circuit serially per rank. For fewer/larger subcircuits use Regime B (below).
#
# Submit with:  sbatch qcut_mpi.sbatch
#
#SBATCH --job-name=qcut-mpi
#SBATCH --account=project_XXXXXXXXX      # <-- set your LUMI project
#SBATCH --partition=standard             # LUMI-C CPU partition
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=128            # 128 ranks/node = 1 core/rank
#SBATCH --cpus-per-task=1
#SBATCH --mem=0                          # use all node memory
#SBATCH --time=01:00:00

module load LUMI/24.03 partition/C
module load cray-python                  # Cray-mpich-linked python + mpi4py

# One thread per rank; stop Aer/NumPy-BLAS from spawning extra threads (the classic
# OpenMP x MPI oversubscription bug).
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_PLACES=cores
export OMP_PROC_BIND=close

# Point this at your QCut virtualenv if not using a module-installed QCut.
# source /path/to/venv/bin/activate

srun python -u run_qcut_job.py

# ---------------------------------------------------------------------------
# Regime B (fewer, larger circuits): give each rank several Aer threads.
#   #SBATCH --ntasks-per-node=8
#   #SBATCH --cpus-per-task=16           # 8 x 16 = 128
#   export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}   # = 16
# QCut.mpi_run reads OMP_NUM_THREADS and sets Aer's max_parallel_threads to match.
# ---------------------------------------------------------------------------
