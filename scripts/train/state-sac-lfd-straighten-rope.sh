env=RopeFlatten
seed=11
now=$(date +%m.%d.%H.%M)

python experiments/run_curl.py \
--env_name=${env} \
--name=SOTA_${env}_SAC_${now}_${seed} \
--log_dir=./data/sota_${env}_sac_${now}_${seed} \
--env_kwargs_observation_mode=key_point \
--env_kwargs_num_variations=1000 \
--expert_data=data/RopeFlatten_numvariations1000_eps10000_trajs_v3.pkl \
--num_train_steps=3_100_000 \
--seed=${seed}
