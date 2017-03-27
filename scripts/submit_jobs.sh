#!/bin/bash

#

#SBATCH --partition=batch

#SBATCH --job-name=styletransfer

#SBATCH --output=output-%A.txt

#SBATCH --error=error-%A.txt

#SBATCH --mem=12G

#SBATCH --cpus-per-task=32

#SBATCH --ntasks=1

srun --unbuffered python styletransfer.py
