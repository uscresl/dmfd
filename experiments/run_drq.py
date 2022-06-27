import time
import torch
import click
import socket
# from chester.run_exp import run_experiment_lite, VariantGenerator
from softgym.registered_env import env_arg_dict
from drq.train import run_task
from datetime import datetime
from sb3.utils import str2bool

reward_scales = {
    'PourWater': 20.0,
    'PassWater': 20.0,
    'ClothFold': 50.0,
    'ClothFoldRobot': 50.0,
    'ClothFoldRobotHard': 50.0,
    'ClothFlatten': 50.0,
    'ClothDrop': 50.0,
    'RopeFlatten': 50.0,
}

clip_obs = {
    'PassWater': None,
    'PourWater': None,
    'ClothFold': None,  # (-3, 3),
    'ClothFoldRobot': None,
    'ClothFoldRobotHard': None,
    'ClothFlatten': None,  # (-2, 2),
    'ClothDrop': None,
    'RopeFlatten': None,
}


def get_critic_lr(env_name, obs_mode):
    return 1e-3


def get_alpha_lr(env_name, obs_mode):
    return 1e-3


def get_lr_decay(env_name, obs_mode):
    return None


def main():
    import argparse
    parser = argparse.ArgumentParser()

    # Experiment
    parser.add_argument('--name', default='CURL_DrQ', type=str)
    parser.add_argument('--env_name', default='RopeFlatten')
    parser.add_argument('--log_dir', default='./data/drq/')
    parser.add_argument('--test_episodes', default=10, type=int)
    parser.add_argument('--seed', default=100, type=int)
    parser.add_argument('--log_save_tb', default=True)  # Save stats to tensorbard
    parser.add_argument('--save_video', default=False)
    parser.add_argument('--save_model', default=True)  # Save trained models
    parser.add_argument('--log_interval', default=10000)  # Save trained models

    # evaluation arguments
    parser.add_argument('--is_eval',        default=False, type=str2bool, help="evaluation or training mode")
    parser.add_argument('--checkpoint',     default=None, type=str, help="actor checkpoint file for evaluation")
    parser.add_argument('--eval_over_five_seeds', default=False, type=str2bool, help="evaluation over 5 random seeds (100 episodes per seed)")

    # Drq
    parser.add_argument('--alpha_fixed', default=False, type=bool)  # Automatic tuning of alpha
    parser.add_argument('--init_temperature', default=0.1, type=float)
    parser.add_argument('--replay_buffer_capacity', default=100000)
    parser.add_argument('--batch_size', default=128)
    parser.add_argument('--num_train_steps', default=1000000, type=int)
    parser.add_argument('--im_size', default=128)

    # Override environment arguments
    parser.add_argument('--env_kwargs_render', default=True, type=bool)  # Turn off rendering can speed up training
    parser.add_argument('--env_kwargs_camera_name', default='default_camera', type=str)
    parser.add_argument('--env_kwargs_observation_mode', default='cam_rgb', type=str)  # Should be in ['key_point', 'cam_rgb', 'point_cloud']
    parser.add_argument('--env_kwargs_deterministic', default=False, type=bool)  # Should be in ['key_point', 'cam_rgb', 'point_cloud']
    parser.add_argument('--env_kwargs_num_variations', default=1000, type=int)

    parser.add_argument('--wandb', action='store_true', help="use wandb instead of tensorboard for logging")

    args = parser.parse_args()

    args.algorithm = 'Drq'

    # Set env_specific parameters
    env_name = args.env_name
    obs_mode = args.env_kwargs_observation_mode
    args.actor_lr = args.critic_lr = get_critic_lr(env_name, obs_mode)
    args.lr_decay = get_lr_decay(env_name, obs_mode)
    args.scale_reward = reward_scales[env_name]
    args.clip_obs = clip_obs[env_name] if obs_mode == 'key_point' else None
    args.env_kwargs = env_arg_dict[env_name]
    now = datetime.now().strftime("%m.%d.%H.%M")
    args.name = f'{env_name}_SAC-DrQ_{now}' if not args.name else args.name
    args.log_dir=f'{args.log_dir}/{args.name}'

    run_task(args.__dict__, args.log_dir, args.name)


if __name__ == '__main__':
    main()
