#!/usr/bin/env bash
#SBATCH --gres=gpu:1
#SBATCH -N1
#SBATCH -n1
#SBATCH -c8
#SBATCH --output=tmp/softgym-%j.log

seed=$1
env=$2
now=$(date +%m.%d.%H.%M)

#Set numvars to 1 for clothfoldRobot
if [[ $env == "ClothFoldRobot"* ]]
  then
    num_variations=1
else
    num_variations=1000
fi

# Set RSI file
# WARNING: Only for clothfoldrobot and ropeflatten
if [[ $env == "ClothFoldRobot"* ]]
  then
    rsi_file=data/ClothFoldRobot_numvariations1_eps1000_image_based_trajs_v2.pkl
else
    rsi_file=data/RopeFlatten_numvariations1000_eps10000_trajs_v3.pkl
fi


python experiments/run_sb3.py \
--env_name=${env} \
--env_kwargs_observation_mode=key_point \
--env_kwargs_num_variations=${num_variations} \
--agent=awac \
--non_rsi_ir=False \
--enable_normalize_obs=False \
--enable_action_matching=False \
--enable_stack_frames=False \
--rsi_file=${rsi_file} \
--add_sac_loss=False \
--sac_loss_weight=0.0 \
--val_freq=10000 \
--val_num_eps=10 \
--sb3_iterations=3_000_000 \
--enable_rsi=False \
--rsi_ir_prob=0.0 \
--name=SOTA_${env}_sb3_awac_baseline_${now}_STATE${seed} \
--seed=${seed} \
--wandb \
