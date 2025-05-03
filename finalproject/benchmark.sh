#!/bin/bash

# set output and error output filenames, %j will be replaced by Slurm with the jobid
#SBATCH -o testing%j.out
#SBATCH -e testing%j.err 

# single node in the "short" partition
#SBATCH -N 1
#SBATCH -p gpu

#SBATCH --mail-user=s.borad@gwu.edu
#SBATCH --mail-type=ALL

# half hour timelimit
#SBATCH -t 24:00:00

module load python3/3.12.9_sqlite

source .venvs/benchmark/bin/activate

python3 benchmark_cls.py