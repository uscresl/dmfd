env=ClothFoldRobot # ['ClothFoldRobot']
seed=11
now=$(date +%m.%d.%H.%M)

python experiments/run_sb3.py \
--env_name=${env} \
--name=SOTA_${env}_DMfD_${now}_${seed} \
--env_kwargs_observation_mode=cam_rgb_key_point \
--agent=awac \
--non_rsi_ir=False \
--enable_normalize_obs=False \
--enable_action_matching=False \
--enable_stack_frames=False \
--awac_replay_size=600_000 \
--rsi_file=data/ClothFoldRobot_numvariations1_eps1000_image_based_trajs_v2.pkl \
--batch_size=256 \
--val_num_eps=10 \
--add_sac_loss=True \
--sac_loss_weight=0.1 \
--enable_rsi=True \
--rsi_ir_prob=0.3 \
--enable_img_aug=True \
--enable_drq_loss=False \
--sb3_iterations=1_000_000 \
--seed=${seed} \
--enable_loading_states_from_folder=True \
--env_kwargs_num_picker=1 \
--env_kwargs_num_variations=1
