#!/bin/bash
#
#SBATCH --job-name=colepatten
#SBATCH --nodes=1
#SBATCH --time=05:00:00
#SBATCH --ntasks-per-node=40
#SBATCH --output=out-%j.log
#SBATCH --partition=bigmem
##SBATCH --ntasks=100
#SBATCH --mail-type=ALL
#SBATCH --mail-user=cole.patten@jacks.sdstate.edu
#SBATCH --gres=gpu:0

module load tensorflow
python3 import_images.py
