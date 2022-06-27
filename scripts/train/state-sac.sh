env=RopeFlatten # ['RopeFlatten', 'ClothFold']
seed=11
now=$(date +%m.%d.%H.%M)
env_kwargs_num_variations=1000

python experiments/run_curl.py \
--env_name=${env} \
--name=SOTA_${env}_SAC_${now}_${seed} \
--log_dir=./data/sota_sac_${env} \
--env_kwargs_observation_mode=key_point \
--env_kwargs_num_variations=${env_kwargs_num_variations} \
--num_train_steps=3_010_000 \
--seed=${seed}
