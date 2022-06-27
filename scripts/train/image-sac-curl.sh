# ClothFoldRobot is ClothFoldDiagonalPinned.
# ClothFoldRobotHard is ClothFoldDiagonalUnpinned.

env=RopeFlatten # ['RopeFlatten', 'ClothFold', 'ClothFoldRobot', 'ClothFoldRobotHard']
seed=11
now=$(date +%m.%d.%H.%M)
env_kwargs_num_variations=1000 # 1000 for RopeFlatten and ClothFold ; 1 for ClothFoldRobot and ClothFoldRobotHard

python experiments/run_curl.py \
--env_kwargs_observation_mode=cam_rgb \
--env_kwargs_num_variations=${env_kwargs_num_variations} \
--name=SAC_CURL_${env} \
--env_name=${env} \
--log_dir=./data/sota_curl_${env} \
--seed=${seed}
