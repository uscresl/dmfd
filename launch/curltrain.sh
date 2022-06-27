#!/usr/bin/env bash
#SBATCH --gres=gpu:1
#SBATCH -N1
#SBATCH -n1
#SBATCH -c16
#SBATCH --output=tmp/softgym-%j.log

# Process seed
if [[ $# -eq 0 ]]
  then
    seed=11
    echo "No seed specified. Using $seed"
else
    seed=$1
fi
now=$(date +%m.%d.%H.%M)
env=$2

if [[ $env == "ClothFoldRobot"* ]]
  then
    num_variations=1
else
    num_variations=1000
fi

srun \
python experiments/run_curl.py \
--env_name=${env} \
--name=SOTA_${env}_SACCURL_${now}_IMAGE${seed} \
--log_dir=./data/sota_curl_${env} \
--env_kwargs_observation_mode=cam_rgb \
--env_kwargs_num_variations=${num_variations} \
--seed=${seed} \
--wandb \
