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
rsiprob=0.0
now=$(date +%m.%d.%H.%M)
env=$2

srun \
python experiments/run_sb3.py \
--env_name=${env} \
--env_kwargs_observation_mode=cam_rgb_key_point \
--env_kwargs_num_variations=1000 \
--agent=awac \
--non_rsi_ir=False \
--enable_normalize_obs=False \
--enable_action_matching=False \
--enable_stack_frames=False \
--awac_replay_size=600_000 \
--rsi_file=data/${env}_numvariations1000_eps8000_replaybuf600000_32x32_trajs_v3.pkl \
--batch_size=256 \
--val_freq=10000 \
--val_num_eps=10 \
--add_sac_loss=False \
--sac_loss_weight=0.0 \
--enable_rsi=False \
--rsi_ir_prob=${rsiprob} \
--enable_img_aug=False \
--enable_drq_loss=False \
--sb3_iterations=1_000_000 \
--name=ABL-${env}_AWAC_${now}_ImgBaseline_${seed} \
--seed=${seed} \
--wandb \
