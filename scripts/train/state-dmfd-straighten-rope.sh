env=RopeFlatten # ['RopeFlatten']
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
--rsi_file=data/RopeFlatten_numvariations1000_eps10000_trajs_v3.pkl \
--add_sac_loss=True \
--sac_loss_weight=0.1 \
--val_freq=10000 \
--val_num_eps=10 \
--sb3_iterations=3_000_000 \
--enable_rsi=True \
--rsi_ir_prob=0.2 \
--seed=${seed}
