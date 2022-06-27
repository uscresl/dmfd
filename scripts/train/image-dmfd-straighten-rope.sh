env=RopeFlatten # ['RopeFlatten']
seed=11
now=$(date +%m.%d.%H.%M)

python experiments/run_sb3.py \
--env_name=${env} \
--name=SOTA_${env}_DMfD_${now}_${seed} \
--env_kwargs_observation_mode=cam_rgb_key_point \
--env_kwargs_num_variations=1000 \
--agent=awac \
--non_rsi_ir=False \
--enable_normalize_obs=False \
--enable_action_matching=False \
--enable_stack_frames=False \
--awac_replay_size=600_000 \
--rsi_file=data/RopeFlatten_numvariations1000_eps8000_replaybuf600000_32x32_trajs_v3.pkl \
--batch_size=256 \
--val_freq=10000 \
--val_num_eps=10 \
--add_sac_loss=True \
--sac_loss_weight=0.1 \
--enable_rsi=True \
--rsi_ir_prob=0.3 \
--enable_img_aug=True \
--enable_drq_loss=False \
--sb3_iterations=1_000_000 \
--seed=${seed}
