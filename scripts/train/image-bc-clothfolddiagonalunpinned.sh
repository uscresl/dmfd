env=ClothFoldRobotHard # ['RopeFlatten']
seed=11

python experiments/run_bc.py \
--env_name=${env} \
--env_img_size=32 \
--num_eval_eps=10 \
--seed=${seed} \
--load_ob_image_mode=direct \
--env_kwargs_num_variations=1 \
--max_train_epochs=1000 \
--is_image_based=True \
--saved_rollouts=data/ClothFoldRobotHard_numvariations1_eps1000_image_based_trajs.pkl \
--action_size=4
