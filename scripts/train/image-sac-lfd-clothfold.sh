env=ClothFold
seed=11
now=$(date +%m.%d.%H.%M)

python experiments/run_asym_sac.py \
--env_name=${env} \
--name=SOTA_${env}_Asym_SAC_PrefilledRepBuf_${now}_${seed} \
--log_dir=./data/sota_${env}_asym_sac_prefilledrepbuf_${now}_${seed} \
--env_kwargs_observation_mode=cam_rgb_key_point \
--env_kwargs_num_variations=1000 \
--num_train_steps=3_020_000 \
--expert_data=data/ClothFold_numvariations1000_eps6000_image_based_replaybuf600000_trajs_v3.pkl \
--seed=${seed}
