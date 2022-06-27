#!/usr/bin/env bash
#SBATCH --gres=gpu:1
#SBATCH -N1
#SBATCH -n1
#SBATCH -c16
#SBATCH --output=tmp/softgym-%j.log
seeds=$1
now=$(date +%m.%d.%H.%M)
env=$2
if [[ $env == "ClothFoldRobot"* ]]
  then
    num_variations=1
else
    num_variations=1000
fi

srun \
parallel \
python experiments/run_curl.py \
--env_name=${env} \
--name=SOTA_${env}_SAC_${now}_{} \
--log_dir=./data/sota_sac_${env} \
--env_kwargs_observation_mode=key_point \
--env_kwargs_num_variations=${num_variations} \
--num_train_steps=1_100_000 \
--seed={} \
--wandb \
::: $seeds \
