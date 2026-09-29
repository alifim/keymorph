#!/bin/bash
#
#SBATCH --job-name=PreprocessAbdominalSeg
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=72:00:00
#SBATCH --mem=16GB
#SBATCH -p sablab-cpu
#SBATCH --output=/midtier/sablab/scratch/alm4065/keymorph/sbatch_scripts/logs/preprocess_%j.out
#SBATCH --error=/midtier/sablab/scratch/alm4065/keymorph/sbatch_scripts/logs/preprocess_%j.err

set -e
module purge
module load anaconda3
source /midtier/sablab/scratch/alm4065/keymorph/.venv/bin/activate
python3 /midtier/sablab/scratch/alm4065/keymorph/scripts/preprocess_abdominal_segmentation.py
