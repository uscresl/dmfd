env=RopeFlatten
seed=11
now=$(date +%m.%d.%H.%M)

python experiments/run_curl.py \
--env_name=${env} \
--name=eval_folder \
--log_dir=./data/eval_folder \
--env_kwargs_observation_mode=key_point \
--env_kwargs_num_variations=1000 \
--is_eval=True \
--checkpoint=checkpoints/RopeFlatten/sac-lfd-state/SOTA_RopeFlatten_SAC_04.16.15.10_11_3000000.pt \
--eval_over_five_seeds=True \
--seed=${seed}