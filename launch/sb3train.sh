#!/usr/bin/env bash
#SBATCH --gres=gpu:1
#SBATCH -N1
#SBATCH -n1
#SBATCH -c8
#SBATCH --output=tmp/softgym-%j.log

# Process seed
if [[ $# -eq 0 ]]
  then
    seed=11
    echo "No seed specified. Using $seed"
else
    seed=$1
fi
env=$2
rsiprob=$3
now=$(date +%m.%d.%H.%M)

if [[ $env == "ClothFoldRobot"* ]]
  then
    num_variations=1
else
    num_variations=1000
fi

srun \
python experiments/run_sb3.py \
--env_name=${env} \
--env_kwargs_observation_mode=cam_rgb_key_point \
--env_kwargs_num_variations=${num_variations} \
--agent=awac \
--non_rsi_ir=False \
--enable_normalize_obs=False \
--enable_action_matching=False \
--enable_stack_frames=False \
--awac_replay_size=600_000 \
--rsi_file=data/ClothFoldRobot_numvariations1_eps1000_image_based_trajs_v2.pkl \
--enable_loading_states_from_folder=True \
--batch_size=256 \
--val_freq=10000 \
--val_num_eps=10 \
--add_sac_loss=True \
--sac_loss_weight=0.1 \
--enable_rsi=True \
--rsi_ir_prob=${rsiprob} \
--enable_img_aug=True \
--enable_drq_loss=False \
--sb3_iterations=1_000_000 \
--name=SOTA_${env}_AWAC-RSI${rsiprob}_${now}_IMAGE${seed} \
--seed=${seed} \
--wandb \

# --env_name=ClothFoldRobot \
# --rsi_ir_prob=0.3 \
# --seed=33 \
# --env_kwargs_num_variations=1 \