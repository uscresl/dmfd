env=ClothFoldRobotHard
seed=11
now=$(date +%m.%d.%H.%M)

python experiments/run_asym_sac.py \
--env_name=${env} \
--name=SOTA_${env}_Asym_SAC_PrefilledRepBuf_${now}_${seed} \
--log_dir=./data/sota_${env}_asym_sac_prefilledrepbuf_${now}_${seed} \
--env_kwargs_observation_mode=cam_rgb_key_point \
--env_kwargs_num_variations=1 \
--num_train_steps=1_100_000 \
--expert_data=data/ClothFoldRobotHard_numvariations1_eps1000_image_based_trajs.pkl \
--seed=${seed}
