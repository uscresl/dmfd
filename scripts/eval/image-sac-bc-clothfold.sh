env=ClothFold
seed=11
now=$(date +%m.%d.%H.%M)

CUDA_VISIBLE_DEVICES=1 python experiments/run_asym_sac.py \
--env_name=${env} \
--name=eval_folder \
--log_dir=./data/eval_folder \
--env_kwargs_observation_mode=cam_rgb_key_point \
--env_kwargs_num_variations=1000 \
--is_eval=True \
--checkpoint=checkpoints/ClothFold/sac-bc-image/SOTA_ClothFold_Asym_SAC_PretrainedBC_05.02.13.25_11_3000000.pt \
--eval_over_five_seeds=True \
--seed=${seed}