env=ClothFold # ['ClothFold']
seed=11

python experiments/run_bc.py \
--env_name=${env} \
--env_img_size=32 \
--num_eval_eps=10 \
--seed=${seed} \
--load_ob_image_mode=direct \
--env_kwargs_num_variations=1000 \
--max_train_epochs=1000 \
--is_image_based=True \
--saved_rollouts=data/ClothFold_numvariations1000_eps6000_image_based_replaybuf600000_trajs_v3.pkl
