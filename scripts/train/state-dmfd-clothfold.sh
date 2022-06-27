env=ClothFold # ['ClothFold']
seed=11
now=$(date +%m.%d.%H.%M)

python experiments/run_sb3.py \
--env_name=${env} \
--name=SOTA_${env}_DMfD_${now}_${seed} \
--env_kwargs_observation_mode=key_point \
--env_kwargs_num_variations=1000 \
--agent=awac \
--non_rsi_ir=False \
--enable_normalize_obs=False \
--enable_action_matching=False \
--enable_stack_frames=False \
--rsi_file=data/ClothFold_numvariations1000_eps10000_state_based_trajs.pkl \
--add_sac_loss=True \
--sac_loss_weight=0.1 \
--enable_rsi=True \
--rsi_ir_prob=0.2 \
--sb3_iterations=3_000_000 \
--seed=55 \
--enable_loading_states_from_folder=True
