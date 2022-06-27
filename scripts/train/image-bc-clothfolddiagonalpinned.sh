env=ClothFoldRobot # ['ClothFoldRobot']
seed=${seed}

python experiments/run_bc.py \
--env_name=${env} \
--env_img_size=32 \
--num_eval_eps=10 \
--seed=11 \
--load_ob_image_mode=direct \
--env_kwargs_num_variations=1 \
--max_train_epochs=1000 \
--is_image_based=True \
--saved_rollouts=data/ClothFoldRobot_numvariations1_eps1000_image_based_trajs_v2.pkl \
--action_size=4
