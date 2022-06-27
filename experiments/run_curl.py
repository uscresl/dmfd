from curl.train import run_task
from softgym.registered_env import env_arg_dict
from sb3.utils import str2bool
from datetime import datetime

reward_scales = {
    'PassWater': 20.0,
    'PourWater': 20.0,
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
    'ClothFold': (-3, 3),
    'ClothFoldRobot': (-3, 3),
    'ClothFoldRobotHard': (-3, 3),
    'ClothFlatten': (-2, 2),
    'ClothDrop': None,
    'RopeFlatten': None,
}


def get_lr_decay(env_name, obs_mode):
    if env_name == 'RopeFlatten' or (env_name == 'ClothFlatten' and obs_mode == 'cam_rgb'):
        return 0.01
    elif obs_mode == 'point_cloud':
        return 0.01
    else:
        return None


def get_actor_critic_lr(env_name, obs_mode):
    if env_name == 'ClothFold' or (env_name == 'RopeFlatten' and obs_mode == 'point_cloud'):
        if obs_mode == 'cam_rgb':
            return 1e-4
        else:
            return 5e-4
    if obs_mode == 'cam_rgb':
        return 3e-4
    else:
        return 1e-3


def get_alpha_lr(env_name, obs_mode):
    if env_name == 'ClothFold':
        return 2e-5
    else:
        return 1e-3


def main():
    import argparse
    parser = argparse.ArgumentParser()
    # Experiment
    parser.add_argument('--name', default='CURL_SAC_ROPE', type=str)
    parser.add_argument('--env_name', default='RopeFlatten')
    parser.add_argument('--log_dir', default='./data/curl_ropeflatten/')
    parser.add_argument('--test_episodes', default=10, type=int)
    parser.add_argument('--num_train_steps', default=1_100_000, type=int, help='Number of training steps')
    parser.add_argument('--seed', default=100, type=int)
    parser.add_argument('--save_tb', default=True)  # Save stats to tensorbard
    parser.add_argument('--save_video', default=True)
    parser.add_argument('--save_model', default=True)  # Save trained models
    parser.add_argument('--wandb', action='store_true', help="use wandb for logging (syncs tensorboard)")

    # evaluation arguments
    parser.add_argument('--is_eval',        default=False, type=str2bool, help="evaluation or training mode")
    parser.add_argument('--checkpoint',     default=None, type=str, help="actor checkpoint file for evaluation")
    parser.add_argument('--eval_over_five_seeds', default=False, type=str2bool, help="evaluation over 5 random seeds (100 episodes per seed)")

    # CURL
    parser.add_argument('--alpha_fixed', default=False, type=bool)  # Automatic tuning of alpha
    parser.add_argument('--init_temperature', default=0.1, type=float)
    parser.add_argument('--replay_buffer_capacity', default=100000)
    parser.add_argument('--batch_size', default=128)

    ############## RSI+IR ##############
    parser.add_argument('--enable_rsi',     default=False, type=str2bool, help="whether or not reference state initialization (RSI) is enabled")
    parser.add_argument('--rsi_file',       default=None, type=str, help='Reference State Initialization file. Path to the trajectory to imitate')
    parser.add_argument('--rsi_ir_prob',    default=0.1, type=float, help='RSI+IR with probability x')
    parser.add_argument('--non_rsi_ir',     default=False, type=str2bool, help='whether or not to use non-RSI+IR')

    # Override environment arguments
    parser.add_argument('--env_kwargs_render', default=True, type=bool)  # Turn off rendering can speed up training
    parser.add_argument('--env_kwargs_camera_name', default='default_camera', type=str)
    parser.add_argument('--env_kwargs_observation_mode', default='cam_rgb', type=str)  # Should be in ['key_point', 'cam_rgb', 'point_cloud']
    parser.add_argument('--env_kwargs_num_variations', default=1000, type=int)

    args = parser.parse_args()

    args.algorithm = 'CURL'

    # Set env_specific parameters

    env_name = args.env_name
    obs_mode = args.env_kwargs_observation_mode
    args.actor_lr = args.critic_lr = get_actor_critic_lr(env_name, obs_mode)
    args.lr_decay = get_lr_decay(env_name, obs_mode)
    args.scale_reward = reward_scales[env_name]
    args.clip_obs = clip_obs[env_name] if obs_mode == 'key_point' else None
    args.env_kwargs = env_arg_dict[env_name]

    args.env_kwargs['enable_rsi'] = args.enable_rsi
    args.env_kwargs['rsi_file'] = args.rsi_file
    args.env_kwargs['rsi_ir_prob'] = args.rsi_ir_prob
    args.env_kwargs['non_rsi_ir'] = args.non_rsi_ir
    now = datetime.now().strftime("%m.%d.%H.%M")
    args.name = f'{env_name}_SAC-CURL_{now}' if not args.name else args.name
    args.log_dir=f'{args.log_dir}/{args.name}'

    run_task(args.__dict__, args.log_dir, args.name)


if __name__ == '__main__':
    main()
