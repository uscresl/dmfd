# ClothFoldRobot is ClothFoldDiagonalPinned.
# ClothFoldRobotHard is ClothFoldDiagonalUnpinned.

env=RopeFlatten # ['RopeFlatten', 'ClothFold', 'ClothFoldRobot', 'ClothFoldRobotHard']
seed=11
now=$(date +%m.%d.%H.%M)
env_kwargs_num_variations=1000 # 1000 for RopeFlatten and ClothFold ; 1 for ClothFoldRobot and ClothFoldRobotHard

python experiments/run_drq.py \
--name=SOTA_Drq_SAC_${env} \
--env_name=${env} \
--log_dir=./data/sota_drq_${env} \
--env_kwargs_num_variations=${env_kwargs_num_variations} \
--seed=${seed}
