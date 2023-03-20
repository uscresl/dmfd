import numpy as np
import torch
import os
import time
import json
import copy

from curl import utils
from curl.logger import Logger

from curl.curl_sac import CoLSacAgent
from curl.default_config import DEFAULT_CONFIG

from chester import logger
from envs.env import SoftGymEnvSB3

from softgym.utils.visualization import save_numpy_as_gif, make_grid
import matplotlib.pyplot as plt
import wandb
from tqdm import tqdm

def update_env_kwargs(vv):
    new_vv = vv.copy()
    for v in vv:
        if v.startswith('env_kwargs_'):
            arg_name = v[len('env_kwargs_'):]
            new_vv['env_kwargs'][arg_name] = vv[v]
            del new_vv[v]
    return new_vv


def vv_to_args(vv):
    class VArgs(object):
        def __init__(self, vv):
            for key, val in vv.items():
                setattr(self, key, val)

    args = VArgs(vv)

    # Dump parameters
    with open(os.path.join(logger.get_dir(), 'variant.json'), 'w') as f:
        json.dump(vv, f, indent=2, sort_keys=True)

    return args


def run_task(vv, log_dir=None, exp_name=None):
    if log_dir or logger.get_dir() is None:
        logger.configure(dir=log_dir, exp_name=exp_name, format_strs=['csv'])
    logdir = logger.get_dir()
    assert logdir is not None
    os.makedirs(logdir, exist_ok=True)
    updated_vv = copy.copy(DEFAULT_CONFIG)
    updated_vv.update(**vv)
    if vv['is_eval']:
        if vv['eval_over_five_seeds']:
            main_eval_five_seeds(vv_to_args(updated_vv))
        else:
            main_eval(vv_to_args(updated_vv))
    else:
        main(vv_to_args(updated_vv))


def get_info_stats(infos):
    # infos is a list with N_traj x T entries
    N = len(infos)
    T = len(infos[0])
    stat_dict_all = {key: np.empty([N, T], dtype=np.float32) for key in infos[0][0].keys()}
    for i, info_ep in enumerate(infos):
        for j, info in enumerate(info_ep):
            for key, val in info.items():
                stat_dict_all[key][i, j] = val

    stat_dict = {}
    for key in infos[0][0].keys():
        stat_dict[key + '_mean'] = np.mean(np.array(stat_dict_all[key]))
        stat_dict[key + '_final'] = np.mean(stat_dict_all[key][:, -1])
    return stat_dict

def fill_replay_buffer_with_img_obs(replay_buffer, expert_data):
    expert_data = np.load(expert_data, allow_pickle=True)
    states = expert_data['ob_trajs']
    next_states = expert_data['ob_next_trajs']
    images = expert_data['ob_img_trajs']
    next_images = expert_data['ob_img_next_trajs']
    actions = expert_data['action_trajs']
    rewards = expert_data['reward_trajs']
    dones = expert_data['done_trajs']
    for ep_counter in range(states.shape[0]):
        for traj_counter in range(len(states[ep_counter])):
            obs = {
                'key_point': states[ep_counter][traj_counter],
                'image': images[ep_counter][traj_counter].transpose(2, 0, 1),
            }
            next_obs = {
                'key_point': next_states[ep_counter][traj_counter],
                'image': next_images[ep_counter][traj_counter].transpose(2, 0, 1),
            }
            replay_buffer.add(
                obs,
                actions[ep_counter][traj_counter],
                rewards[ep_counter][traj_counter],
                next_obs,
                dones[ep_counter][traj_counter],
            )

def evaluate(env, agent, video_dir, num_episodes, L, step, args):
    all_ep_rewards = []

    def run_eval_loop(sample_stochastically=True):
        start_time = time.time()
        prefix = 'stochastic_' if sample_stochastically else ''
        infos = []
        all_frames = []
        plt.figure()
        for i in range(num_episodes):
            obs = env.reset()
            done = False
            episode_reward = 0
            ep_info = []
            frames = [env.get_image(128, 128)]
            rewards = []
            while not done:
                with utils.eval_mode(agent):
                    if sample_stochastically:
                        action = agent.sample_action(obs)
                    else:
                        action = agent.select_action(obs)
                obs, reward, done, info = env.step(action)
                episode_reward += reward
                ep_info.append(info)
                frames.append(env.get_image(128, 128))
                rewards.append(reward)
            plt.plot(range(len(rewards)), rewards)
            if len(all_frames) < 8:
                all_frames.append(frames)
            infos.append(ep_info)

            L.log('eval/' + prefix + 'episode_reward', episode_reward, step)
            all_ep_rewards.append(episode_reward)
        plt.savefig(os.path.join(video_dir, '%d.png' % step))
        all_frames = np.array(all_frames).swapaxes(0, 1)
        all_frames = np.array([make_grid(np.array(frame), nrow=2, padding=3) for frame in all_frames])
        save_numpy_as_gif(all_frames, os.path.join(video_dir, '%d.gif' % step))

        for key, val in get_info_stats(infos).items():
            L.log('eval/info_' + prefix + key, val, step)
            if args.wandb:
                wandb.log({f'val/info_{key}': val, 'num_timesteps': step})
        L.log('eval/' + prefix + 'eval_time', time.time() - start_time, step)
        mean_ep_reward = np.mean(all_ep_rewards)
        best_ep_reward = np.max(all_ep_rewards)
        L.log('eval/' + prefix + 'mean_episode_reward', mean_ep_reward, step)
        L.log('eval/' + prefix + 'best_episode_reward', best_ep_reward, step)

    run_eval_loop(sample_stochastically=False)
    L.dump(step)


def make_agent(obs_shape, action_shape, args, device):
    return CoLSacAgent(
        args=args,
        obs_shape=obs_shape,
        action_shape=action_shape,
        device=device,
        hidden_dim=args.hidden_dim,
        discount=args.discount,
        init_temperature=args.init_temperature,
        alpha_lr=args.alpha_lr,
        alpha_beta=args.alpha_beta,
        alpha_fixed=args.alpha_fixed,
        actor_lr=args.actor_lr,
        actor_beta=args.actor_beta,
        actor_log_std_min=args.actor_log_std_min,
        actor_log_std_max=args.actor_log_std_max,
        actor_update_freq=args.actor_update_freq,
        critic_lr=args.critic_lr,
        critic_beta=args.critic_beta,
        critic_tau=args.critic_tau,
        critic_target_update_freq=args.critic_target_update_freq,
        encoder_type=args.encoder_type,
        encoder_feature_dim=args.encoder_feature_dim,
        encoder_lr=args.encoder_lr,
        encoder_tau=args.encoder_tau,
        num_layers=args.num_layers,
        num_filters=args.num_filters,
        log_interval=args.log_interval,
        detach_encoder=args.detach_encoder,
        curl_latent_dim=args.curl_latent_dim
    )


def main_eval_five_seeds(args):
    args.__dict__ = update_env_kwargs(args.__dict__)  # Update env_kwargs

    symbolic = args.env_kwargs['observation_mode'] != 'cam_rgb'
    args.encoder_type = 'identity' if symbolic else 'pixel'

    env = Env(args.env_name, symbolic, args.seed, 200, 1, 8, args.pre_transform_image_size, env_kwargs=args.env_kwargs, normalize_observation=False,
            scale_reward=args.scale_reward, clip_obs=args.clip_obs)
    env.seed(args.seed)

    args.work_dir = logger.get_dir()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    action_shape = env.action_space.shape
    if args.encoder_type == 'pixel':
        obs_shape = (3, args.image_size, args.image_size)
        pre_aug_obs_shape = (3, args.pre_transform_image_size, args.pre_transform_image_size)
    else:
        obs_shape = env.observation_space.shape
        pre_aug_obs_shape = obs_shape

    agent = make_agent(
        obs_shape=obs_shape,
        action_shape=action_shape,
        args=args,
        device=device
    )
    checkpoint_path = args.checkpoint
    agent.load_actor(checkpoint_path)

    total_normalized_perf_final = []
    random_seeds = [100, 201, 302, 403, 504]
    for curr_seed in random_seeds:
        utils.set_seed_everywhere(curr_seed)
        for episode in range(20):
            obs = env.reset()
            done = False
            ep_normalized_perf = []

            while not done:
                if args.encoder_type == 'pixel':
                    obs = utils.center_crop_image(obs, args.image_size)
                with utils.eval_mode(agent):
                    action = agent.select_action(obs)
                obs, reward, done, info = env.step(action)
                ep_normalized_perf.append(info['normalized_performance'])
            total_normalized_perf_final.append(ep_normalized_perf[-1])
            print(f'Ep {episode}  Episode normalized performance final: {ep_normalized_perf[-1]}')

    total_normalized_perf_final = np.array(total_normalized_perf_final)

    ckpt_file_path = checkpoint_path
    npy_file_path = "/".join(ckpt_file_path.split('/')[:-1]) + '/' + ckpt_file_path.split('-ckpt-')[0].split('/')[-1] + '.npy'
    np.save(npy_file_path, total_normalized_perf_final)

    print('!!!!!!! info_normalized_performance_final !!!!!!!')
    print(f'Mean: {np.mean(total_normalized_perf_final):.4f}')
    print(f'Std: {np.std(total_normalized_perf_final):.4f}')
    print(f'Median: {np.median(total_normalized_perf_final):.4f}')
    print(f'25th Percentile: {np.percentile(total_normalized_perf_final, 25):.4f}')
    print(f'75th Percentile: {np.percentile(total_normalized_perf_final, 75):.4f}')


def main_eval(args):
    if args.seed == -1:
        args.__dict__["seed"] = np.random.randint(1, 1000000)
    utils.set_seed_everywhere(args.seed)

    args.__dict__ = update_env_kwargs(args.__dict__)  # Update env_kwargs

    symbolic = args.env_kwargs['observation_mode'] != 'cam_rgb'
    args.encoder_type = 'identity' if symbolic else 'pixel'

    env = Env(args.env_name, symbolic, args.seed, 200, 1, 8, args.pre_transform_image_size, env_kwargs=args.env_kwargs, normalize_observation=False,
              scale_reward=args.scale_reward, clip_obs=args.clip_obs)
    env.seed(args.seed)

    args.work_dir = logger.get_dir()
    video_dir = utils.make_dir(os.path.join(args.work_dir, 'video'))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    action_shape = env.action_space.shape

    if args.encoder_type == 'pixel':
        obs_shape = (3, args.image_size, args.image_size)
        pre_aug_obs_shape = (3, args.pre_transform_image_size, args.pre_transform_image_size)
    else:
        obs_shape = env.observation_space.shape
        pre_aug_obs_shape = obs_shape

    agent = make_agent(
        obs_shape=obs_shape,
        action_shape=action_shape,
        args=args,
        device=device
    )

    agent.load_actor(args.checkpoint)

    for episode in range(args.num_eval_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        episode_step = 0
        ep_info = []
        frames = [env.get_image(1024, 1024)]
        rewards = []
        ep_normalized_perf = []

        while not done:
            if args.encoder_type == 'pixel':
                obs = utils.center_crop_image(obs, args.image_size)
            with utils.eval_mode(agent):
                action = agent.select_action(obs)
            obs, reward, done, info = env.step(action)
            episode_reward += reward
            episode_step += 1
            ep_info.append(info)
            frames.append(env.get_image(1024, 1024))
            rewards.append(reward)
            ep_normalized_perf.append(info['normalized_performance'])
        save_numpy_as_gif(np.array(frames), os.path.join(video_dir, f'ep_{episode}_{ep_normalized_perf[-1]}.gif'))


def main(args):
    if args.seed == -1:
        args.__dict__["seed"] = np.random.randint(1, 1000000)
    utils.set_seed_everywhere(args.seed)

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
        'image_dim': None if not_imaged_based else 32,
        'env_kwargs': args.env_kwargs,
        'normalize_observation': False,
        'scale_reward': args.scale_reward,
        'clip_obs': args.clip_obs,
        'obs_process': None,
    }
    env = SoftGymEnvSB3(**env_kwargs)

    # make directory
    ts = time.gmtime()
    ts = time.strftime("%m-%d", ts)

    args.work_dir = logger.get_dir()
    # WandB
    if args.wandb:
        wandb_logger = wandb.init(
            project="dmfd",
            config=args.__dict__,
            sync_tensorboard=True,  # auto-upload tensorboard metrics
            name=args.name)

    video_dir = utils.make_dir(os.path.join(args.work_dir, 'video'))
    model_dir = utils.make_dir(os.path.join(args.work_dir, 'model'))
    buffer_dir = utils.make_dir(os.path.join(args.work_dir, 'buffer'))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    action_shape = env.action_space.shape

    keypoint_shape = env.observation_space['key_point'].shape
    img_shape = env.observation_space['image'].shape

    replay_buffer = utils.CoLReplayBuffer(
        keypoint_shape=keypoint_shape,
        img_shape=img_shape,
        action_shape=action_shape,
        capacity=args.replay_buffer_capacity,
        batch_size=int(args.batch_size * 0.75),
        device=device,
        image_size=args.image_size,
    )

    # create a second replay buffer and fill it with expert demonstrations
    expert_replay_buffer = utils.CoLReplayBuffer(
        keypoint_shape=keypoint_shape,
        img_shape=img_shape,
        action_shape=action_shape,
        capacity=args.replay_buffer_capacity,
        batch_size=int(args.batch_size * 0.25),
        device=device,
        image_size=args.image_size,
    )
    fill_replay_buffer_with_img_obs(expert_replay_buffer, args.expert_data)

    agent = make_agent(
        obs_shape=keypoint_shape,
        action_shape=action_shape,
        args=args,
        device=device
    )

    L = Logger(args.work_dir, use_tb=args.save_tb, chester_logger=logger)

    # Pre-training stage
    print('Start pre-training stage...')
    for step in tqdm(range(args.num_pretrain_steps)):
        agent.pretrain_update(expert_replay_buffer, L, step, args.batch_size)

    # Collect some experiences (prevent empty replay buffer bug)
    print('Collecting experiences...')
    episode, episode_reward, done = 0, 0, True
    for _ in tqdm(range(300)):
        if done:
            obs = env.reset()
            done = False
            ep_info = []
            episode_reward = 0
            episode_step = 0
            episode += 1

        with utils.eval_mode(agent):
            action = agent.sample_action(obs)

        next_obs, reward, done, info = env.step(action)

        # allow infinit bootstrap
        done_bool = 0 if episode_step + 1 == env.horizon else float(done)
        episode_reward += reward
        replay_buffer.add(obs, action, reward, next_obs, done_bool)
        obs = next_obs
        episode_step += 1

    # Main training stage
    print('Start main training stage...')
    episode, episode_reward, done, ep_info = 0, 0, True, []
    start_time = time.time()
    for step in range(args.num_pretrain_steps, args.num_train_steps):

        # evaluate agent periodically
        if step % args.eval_freq == 0:
            L.log('eval/episode', episode, step)
            evaluate(env, agent, video_dir, args.num_eval_episodes, L, step, args)
            if args.save_model and (step % (args.eval_freq * 5) == 0):
                agent.save(model_dir, step)
            if args.save_buffer:
                replay_buffer.save(buffer_dir)

        # data collection steps
        for _ in range(args.num_data_collections):
            if done:
                if step > args.num_pretrain_steps:
                    if step % args.log_interval == 0:
                        L.log('train/duration', time.time() - start_time, step)
                        for key, val in get_info_stats([ep_info]).items():
                            L.log('train/info_' + key, val, step)
                        L.dump(step)
                    start_time = time.time()
                if step % args.log_interval == 0:
                    L.log('train/episode_reward', episode_reward, step)

                obs = env.reset()
                done = False
                ep_info = []
                episode_reward = 0
                episode_step = 0
                episode += 1
                if step % args.log_interval == 0:
                    L.log('train/episode', episode, step)

            with utils.eval_mode(agent):
                action = agent.sample_action(obs)

            next_obs, reward, done, info = env.step(action)

            # allow infinit bootstrap
            ep_info.append(info)
            done_bool = 0 if episode_step + 1 == env.horizon else float(done)
            episode_reward += reward
            replay_buffer.add(obs, action, reward, next_obs, done_bool)
            obs = next_obs
            episode_step += 1

         # run training update
        agent.update(replay_buffer, expert_replay_buffer, L, step)

    if args.wandb:
        wandb_logger.finish()
    print('Finished training...')
