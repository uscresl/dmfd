env=RopeFlatten
seed=11
now=$(date +%m.%d.%H.%M)

python experiments/run_asym_sac.py \
--env_name=${env} \
--name=eval_folder \
--log_dir=./data/eval_folder \
--env_kwargs_observation_mode=cam_rgb_key_point \
--env_kwargs_num_variations=1000 \
--is_eval=True \
--checkpoint=checkpoints/RopeFlatten/sac-lfd-image/SOTA_RopeFlatten_Asym_SAC_PrefilledRepBuf_04.20.21.37_11_1000000.pt \
--eval_over_five_seeds=True \
--seed=${seed}