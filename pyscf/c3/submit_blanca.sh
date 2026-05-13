#!/bin/bash
#SBATCH --job-name c3_rhf
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=28
#SBATCH --mem=128000M
#SBATCH --export=NONE
#SBATCH --output=sbatch_blanca.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zhangych98@gmail.com

ml anaconda
source ~/tools/load_modules/load_lno_pyscf_afqmc.sh
export MKL_NUM_THREADS=28
export OMP_NUM_THREADS=28
export OPENBLAS_NUM_THREADS=28
export OMP_NUM_THREADS=28

python c3_rhf.py >> c3_rhf.out
