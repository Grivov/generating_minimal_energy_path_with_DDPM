#!/bin/bash

#SBATCH --job-name=ala
#SBATCH --partition=parallel
#SBATCH --time=3-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
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

#echo -e "1\n1\n" | gmx_mpi pdb2gmx -f protein.pdb -o protein.gro
#gmx_mpi editconf -f protein.gro -o protein_newbox.gro -c -d 1.0 -bt cubic
#gmx_mpi solvate -cp protein_newbox.gro -cs a99SBdisp_water.gro -o protein_solv.gro -p topol.top

#gmx_mpi grompp -f ions.mdp -c protein_solv.gro -p topol.top -o ions.tpr
#echo -e "13" | gmx_mpi genion -s ions.tpr -o protein_solv_ions.gro -p topol.top -pname NA -nname CL -neutral

#gmx_mpi grompp -f minim.mdp -c protein_solv_ions.gro -p topol.top -o em.tpr
#mpirun -np 48 gmx_mpi mdrun -deffnm em

#gmx_mpi grompp -f nvt.mdp -c em.gro -r em.gro -p topol.top -o nvt.tpr
#mpirun -np 48 gmx_mpi mdrun -deffnm nvt

#gmx_mpi grompp -f npt.mdp -c nvt.gro -r nvt.gro -t nvt.cpt -p topol.top -o npt.tpr
#mpirun -np 48 gmx_mpi mdrun -deffnm npt

#gmx_mpi grompp -f md.mdp -c nvt.gro -t nvt.cpt -p topol.top -o md_0_1.tpr
#mpirun -np 48 gmx_mpi mdrun -deffnm md_0_1


echo "--- converting pdb file ---"
gmx_mpi pdb2gmx -f ala2.pdb -water tip3p -ff amber03 || exit 1 #also try amber99

echo "--- setting box size ---"
gmx_mpi editconf -o box.gro -f conf.gro -bt cubic -d 1.2 || exit 1

echo "--- creating water molecules ---"
gmx_mpi solvate -cp box.gro -cs spc216.gro -o sol.gro -p topol.top || exit 1

echo "--- adding NaCl in physiologial concentration ---"
gmx_mpi grompp -o iongen.tpr -c sol.gro -f em.mdp || exit 1
echo "13\n" | gmx_mpi genion -o ionized.gro -s iongen.tpr -p topol.top -conc 0.15 || exit 1

echo "--- running energy minimization ----"
gmx_mpi grompp -o em.tpr -f em.mdp -c ionized.gro || exit 1
mpirun -np 8 gmx_mpi mdrun -deffnm em || exit 1

echo "--- init temperature ----"
gmx_mpi grompp -o nvt.tpr -f nvt.mdp -c em.gro -r em.gro || exit 1
mpirun -np 8 gmx_mpi mdrun -deffnm nvt || exit 1

echo "--- init pressure ----"
gmx_mpi grompp -o npt.tpr -f npt.mdp -c nvt.gro -r nvt.gro || exit 1
mpirun -np 8 gmx_mpi mdrun -deffnm npt || exit 1

#echo "--- final simulation ----"
gmx_mpi grompp -o md.tpr -f md.mdp -c npt.gro -r npt.gro || exit 1
mpirun -np 8 gmx_mpi mdrun -v -deffnm md || exit 1

#echo "--- write phi and psi angles to rama.xvg ---"
#gmx_mpi g_rama -f md.trr -s md.tpr || exit 1

#echo "--- convert trajectory (otherwise, atom is split on borders of box) ---"
#echo "--- also save to xtc instead of trr (smaller file) ---"
#gmx_mpi trjconv -pbc nojump -f md.trr -o md_corr.xtc || exit 1
# rm md.trr || exit 1

echo "SUCCESS, results in:"
echo "md.gro: final configuration"
echo "md_corr.xtc: corrected trajectory"
echo "md.edr: energy for trajectory"
echo "md.cpt: data needed to continue simulation"
echo "rama.xvg: plot of dihedral angles"
