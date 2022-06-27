env=ClothFoldRobotHard
seed=11
now=$(date +%m.%d.%H.%M)

python experiments/run_asym_sac.py \
--env_name=${env} \
--name=eval_folder \
--log_dir=./data/eval_folder \
--env_kwargs_observation_mode=cam_rgb_key_point \
--env_kwargs_num_variations=1 \
--is_eval=True \
--checkpoint=checkpoints/ClothFoldRobotHard/sac-lfd-image/SOTA_ClothFoldRobotHard_Asym_SAC_PrefilledRepBuf_04.22.08.26_11_1000000.pt \
--eval_over_five_seeds=True \
--seed=${seed}