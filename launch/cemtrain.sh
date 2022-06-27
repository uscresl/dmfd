#!/usr/bin/env bash
#SBATCH --gres=gpu:1
#SBATCH -N1
#SBATCH -n1
#SBATCH -c16
#SBATCH --output=tmp/softgym-%j.log

srun python experiments/run_cem.py \
--env_name=ClothFlatten \
--exp_name=cem_clothflatten \
--log_dir=./data/cem_clothflatten
