import joblib
from sb3.train import run_task
from sb3.eval import evaluation
from sb3.utils import str2bool, set_seed_everywhere, update_env_kwargs, make_dir, NumpyEncoder
from softgym.registered_env import env_arg_dict
import torch
import argparse
import json
from datetime import datetime
from stable_baselines3.common.utils import set_random_seed
from awac.AWAC.awac import AWAC

reward_scales = {
    'PassWater': 20.0,
    'PourWater': 20.0,
    'ClothFold': 50.0,
    'ClothFoldRobot': 50.0,
    'ClothFoldRobotHard': 50.0,
    'DryCloth': 50.0,
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
    'DryCloth': (-3, 3),
    'ClothFlatten': (-2, 2),
    'ClothDrop': None,
    'RopeFlatten': None,
}

def main():
    parser = argparse.ArgumentParser()

    ############## Experiment ##############
    # evaluation arguments
    parser.add_argument('--is_eval',        default=False, type=str2bool, help="evaluation or training mode")
    parser.add_argument('--checkpoint',     default=None, type=str, help="checkpoint file for evaluation")
    parser.add_argument('--num_eval_eps',   default=10, type=int, help="number of episodes to run during evaluation")
    parser.add_argument('--eval_videos',    default=False, type=str2bool, help="whether or not to save evaluation video per episode")
    parser.add_argument('--eval_gif_size',  default=256, type=int, help="evaluation GIF width and height size")
    parser.add_argument('--eval_over_five_seeds', default=False, type=str2bool, help="evaluation over 5 random seeds (100 episodes per seed)")

    # validation arguments
    parser.add_argument('--val_freq',       default=10000, type=int, help="validation frequency (env. steps)")
    parser.add_argument('--val_num_eps',    default=10, type=int, help="number of episodes for validation")

    # logging arguments
    parser.add_argument('--wandb',          action='store_true', help="use wandb instead of tensorboard for logging")
    parser.add_argument('--verbose',        type=str2bool, default=False, help="Print info about training and evaluation if set to True")

    # task arguments
    parser.add_argument('--env_name',       default='RopeFlatten')
    parser.add_argument('--image_size',     default=100, type=int, help="center crop image size")

    # SB3 arguments
    parser.add_argument('--agent',          default='td3', choices=['sac', 'td3', 'sac-bc', 'awac'])
    parser.add_argument('--batch_size',     default=256, type=int, help="training batch_size")
    parser.add_argument('--name',           default=None, type=str, help='[optional] set experiment name. Useful to resume experiments.')
    parser.add_argument('--seed',           default=1234, type=int, help="seed number")
    parser.add_argument('--sb3_iterations', default=1_000_000, type=int, help="number of iterations for sb3")

    # This argument is disabled due to the removal of multi-environments (performance issue)
    # parser.add_argument('--debug',          action='store_true', help="enable to use single environment for debugging")
    # parser.add_argument('--num_envs',       default=4, type=int, help="number of environments")

    ############## RSI+IR ##############
    parser.add_argument('--enable_rsi',     default=False, type=str2bool, help="whether or not reference state initialization (RSI) is enabled")
    parser.add_argument('--rsi_file',       default=None, type=str, help='Reference State Initialization file. Path to the trajectory to imitate')
    parser.add_argument('--rsi_ir_prob',    default=0.0, type=float, help='RSI+IR with probability x')
    parser.add_argument('--non_rsi_ir', default=False, type=str2bool, help='whether or not to use non-RSI+IR')
    parser.add_argument('--enable_action_matching', default=False, type=str2bool, help="whether or not action matching is enabled")
    parser.add_argument('--enable_loading_states_from_folder', default=False, type=str2bool, help="whether or not to enable loading states from folder")

    parser.add_argument('--enable_stack_frames', default=False, type=str2bool, help="whether or not to stack 3 frames in an observation")
    parser.add_argument('--enable_normalize_obs', default=False, type=str2bool, help="whether or not to normalize observations to [-1, 1] using running average")
    parser.add_argument('--normalize_obs_file', default=None, type=str, help="file path to normalized observations (max and min)")

    ############## BC Related ##############
    parser.add_argument('--bc_model_ckpt_file', default=None, type=str, help="file path to pretrained BC model checkpoint")
    # use 'auto_0.1' to avoid explording critic loss
    parser.add_argument('--ent_coef', default='auto', type=str, help='Entropy coefficient for SAC')

    ############## AWAC Related ##############
    parser.add_argument('--p_lr', default=3e-4, type=float, help='policy/actor learning rate')
    parser.add_argument('--lr', default=3e-4, type=float, help='critics learning rate')
    parser.add_argument('--awac_replay_size', default=2_000_000, type=int, help="AWAC replay buffer size")
    parser.add_argument('--expert_repeat_num', default=None, type=int, help="Whether or not to repeat expert data. If yes, repeat the expert data expert_repeat_num times")
    parser.add_argument('--add_sac_loss', default=False, type=str2bool, help="whether or not to add SAC's actor loss to AWAC's")
    parser.add_argument('--sac_loss_weight', default=0.0, type=float, help='weight given to sac_loss=sac_loss_weight and awac_loss=(1-sac_loss_weight)')
    parser.add_argument('--critics_input', default='image_state', choices=['image_state', 'image', 'state'])
    parser.add_argument('--enable_img_aug', default=False, type=str2bool, help="whether or not to enable image augmentations")
    parser.add_argument('--enable_drq_loss', default=False, type=str2bool, help="whether or not to use Dr.Q's Q-function")
    parser.add_argument('--save_video_pickplace', action='store_true', default=False, help='Whether to save pick and place recorded videos')
    parser.add_argument('--load_from', default=None, type=str, help="file path to pretrained AWAC object (model, optimizer, etc.)")
    parser.add_argument('--enable_inv_dyn_model', default=False, type=str2bool, help="whether or not to jointly learn DMfD with an inverse dynamics model (inspired by SOIL)")
    parser.add_argument('--inv_dyn_file', default=None, type=str, help='State-only demonstrations for training inverse dynamics model')

    ############## Override environment arguments ##############
    parser.add_argument('--env_kwargs_render', default=True, type=str2bool)  # Turn off rendering can speed up training
    parser.add_argument('--env_kwargs_camera_name', default='default_camera', type=str)
    parser.add_argument('--env_kwargs_observation_mode', default='key_point', type=str)  # Should be in ['key_point', 'cam_rgb', 'point_cloud']. Only AWAC supports 'cam_rgb_key_point' and 'depth_key_point'.
    parser.add_argument('--env_kwargs_num_variations', default=1000, type=int)
    parser.add_argument('--env_kwargs_env_image_size', default=32, type=int, help="observation image size")
    parser.add_argument('--env_kwargs_num_picker', default=2, type=int, help='Number of pickers/end-effectors')
    parser.add_argument('--action_mode', type=str, default=None, help='Overwrite action_mode in the environment')
    parser.add_argument('--action_repeat', type=int, default=None, help='Overwrite action_repeat in the environment')
    parser.add_argument('--horizon', type=int, default=None, help='Overwrite action_repeat in the environment')

    args = parser.parse_args()

    # Set env_specific parameters
    env_name = args.env_name
    obs_mode = args.env_kwargs_observation_mode
    args.scale_reward = reward_scales[env_name]
    args.clip_obs = clip_obs[env_name] if obs_mode == 'key_point' else None
    args.env_kwargs = env_arg_dict[env_name]
    args.__dict__ = update_env_kwargs(args.__dict__)  # Update env_kwargs

    not_imaged_based = args.env_kwargs['observation_mode'] not in ['cam_rgb', 'cam_rgb_key_point', 'depth_key_point']
    symbolic = not_imaged_based
    args.encoder_type = 'identity' if symbolic else 'pixel'
    args.max_steps = 200
    env_kwargs = {
        'env': args.env_name,
        'symbolic': symbolic,
        'seed': args.seed,
        'max_episode_length': args.max_steps,
        'action_repeat': 1,
        'bit_depth': 8,
        'image_dim': None if not_imaged_based else args.env_kwargs['env_image_size'],
        'env_kwargs': args.env_kwargs,
        'normalize_observation': False,
        'scale_reward': args.scale_reward,
        'clip_obs': args.clip_obs,
        'obs_process': None,
    }
    env_kwargs['env_kwargs']['enable_rsi'] = args.enable_rsi
    env_kwargs['env_kwargs']['rsi_file'] = args.rsi_file
    env_kwargs['env_kwargs']['rsi_ir_prob'] = args.rsi_ir_prob
    env_kwargs['env_kwargs']['non_rsi_ir'] = args.non_rsi_ir
    env_kwargs['env_kwargs']['enable_action_matching'] = args.enable_action_matching
    env_kwargs['env_kwargs']['enable_stack_frames'] = args.enable_stack_frames
    env_kwargs['env_kwargs']['enable_normalize_obs'] = args.enable_normalize_obs
    env_kwargs['env_kwargs']['normalize_obs_file'] = args.normalize_obs_file
    env_kwargs['env_kwargs']['enable_loading_states_from_folder'] = args.enable_loading_states_from_folder
    if args.action_mode:
        env_kwargs['env_kwargs']['action_mode'] = args.action_mode
    if args.action_repeat:
        env_kwargs['env_kwargs']['action_repeat'] = args.action_repeat
    if args.horizon:
        env_kwargs['env_kwargs']['horizon'] = args.horizon

    # assertions for various argument combinations
    if args.enable_rsi:
        assert args.rsi_file
    if args.enable_drq_loss:
        assert args.enable_img_aug
    if args.enable_inv_dyn_model:
        assert args.inv_dyn_file

    # get device
    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.cuda.empty_cache()
    else:
        device = torch.device('cpu')
    print(f"Device set to {device}")

    set_random_seed(args.seed)
    set_seed_everywhere(args.seed)

    if args.is_eval:
        if args.agent == 'awac':
            # separate evaluation code for AWAC
            agent = AWAC(args.__dict__, env_kwargs)
            if args.eval_over_five_seeds:
                agent.eval_agent_five_seeds(args.__dict__)
            else:
                agent.eval_agent(args.__dict__)
        else:
            # evaluation code for SB3-based policies
            evaluation(args.__dict__, env_kwargs)
    else:
        torch.set_num_threads(2) # empirically faster in Pendulum environment as shown in https://github.com/DLR-RM/stable-baselines3/issues/90#issuecomment-659607948
        now = datetime.now().strftime("%m.%d.%H.%M")
        args.folder_name = f'{env_name}_SB3_{args.agent}_{now}' if not args.name else args.name
        args.tb_dir = f"./data/sb3/{args.folder_name}"
        args.ckpt_saved_folder = f'{args.tb_dir }/checkpoints/'
        make_dir(f'{args.tb_dir}')
        with open(f'{args.tb_dir}/config.json', 'w') as outfile:
            json.dump(args.__dict__, outfile, indent=2, cls=NumpyEncoder)

        if args.agent == 'awac':
            # separate training code for AWAC
            agent = AWAC(args.__dict__, env_kwargs)
            if args.load_from is not None:
                saved = joblib.load(args.load_from)
                agent.starting_timestep = saved['iterations']
                agent.load_state_dict(saved)
                print(f'Loaded AWAC COMPONENTS from {args.load_from}. Iterations: {saved["iterations"]}')
                print(f'RSI file will be ignored ({args.rsi_file})')
                del saved # free up memory
            else:
                if args.rsi_file:
                    agent.populate_replay_buffer(args.rsi_file, repeat_num=args.expert_repeat_num)
            if args.enable_inv_dyn_model:
                agent.run_with_inv_dyn_model(args)
            else:
                agent.run(args)
        else:
            # training code for SB3-based policies
            run_task(args, env_kwargs)

print(f"Done! Train/eval script finished.")

if __name__ == '__main__':
    main()
