env=RopeFlatten # ['RopeFlatten']
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
--saved_rollouts=data/RopeFlatten_numvariations1000_eps8000_replaybuf600000_32x32_trajs_v3.pkl
