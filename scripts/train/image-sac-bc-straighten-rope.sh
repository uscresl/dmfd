env=RopeFlatten
seed=11
now=$(date +%m.%d.%H.%M)

python experiments/run_asym_sac.py \
--env_name=${env} \
--name=SOTA_${env}_Asym_SAC_PretrainedBC_${now}_${seed} \
--log_dir=./data/sota_${env}_asym_sac_pretrainedbc_${now}_${seed} \
--env_kwargs_observation_mode=cam_rgb_key_point \
--env_kwargs_num_variations=1000 \
--num_train_steps=1_010_000 \
--bc_checkpoint=./checkpoints/RopeFlatten_BC_05.01.13.26/epoch_960.pth \
--seed=${seed}
