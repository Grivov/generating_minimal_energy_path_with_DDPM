#!/bin/bash

#SBATCH --job-name=ala
#SBATCH --partition=shared
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --account=yzhan567
#SBATCH --array=1

module load gcc/9.3.0
module load openmpi/3.1.6
module load gromacs/2020.5
module load hwloc
module load boost
module load plumed/2.3b
export CUDA_AUTO_BOOST=1

#echo "--- convert trajectory (otherwise, atom is split on borders of box) ---"
#echo "--- also save to xtc instead of trr (smaller file) ---"
gmx_mpi trjconv -pbc nojump -f md.trr -o md_corr.xtc || exit 1

gmx_mpi make_ndx -f md_corr.xtc -o index.ndx
echo "0" | gmx_mpi trjconv -s ala2.pdb -f md_corr.xtc -o md_nosol.xtc -n index.ndx
