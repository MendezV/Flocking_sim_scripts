#!/bin/bash
#SBATCH --partition=nodes
#SBATCH --job-name=Flocking_Sim
#SBATCH --mem=1
#SBATCH --time=5-0:0
#SBATCH --output=Flock_res_%a.txt
#SBATCH --array=1-3
#SBATCH --exclude=node30[01-03]


declare -a commands

commands[1]="time python FL_ref.py 10"
commands[2]="time python FL_ref.py 50"
commands[3]="time python FL_ref.py 100"


bash -c "${commands[${SLURM_ARRAY_TASK_ID}]}"
