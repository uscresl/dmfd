import os
import numpy as np
from envs.env import SoftGymEnvSB3
from stable_baselines3 import SAC, TD3
from sb3 import utils
from softgym.utils.visualization import save_numpy_as_gif

def evaluation(config, env_kwargs):
    if config['eval_videos']:
        eval_video_path = utils.make_dir('/'.join(config['checkpoint'].split('/')[:-2]) + '/eval_video')

    env = SoftGymEnvSB3(**env_kwargs)
    if config['agent'] == 'sac':
        model_class = SAC
    elif config['agent'] == 'td3':
        model_class = TD3
    else:
        raise NotImplementedError

    model = model_class.load(config['checkpoint'], env=env)

    total_rewards, total_lengths, total_normalized_perf = 0, 0, []
    num_eval_eps = config['num_eval_eps']
    for ep in range(num_eval_eps):
        obs = env.reset()
        done, ep_len, ep_rew, ep_normalized_perf = False, 0, 0, []
        if config['eval_videos']:
            frames = [env.get_image(config['eval_gif_size'], config['eval_gif_size'])]
        while ep_len < config['max_steps'] and not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            ep_len += 1
            ep_rew += reward
            ep_normalized_perf.append(info['normalized_performance'])
            if config['eval_videos']:
                frames.append(env.get_image(config['eval_gif_size'], config['eval_gif_size']))
        print(f'Current Episode Rewards: {ep_rew}, Episode normalized performance mean: {np.mean(ep_normalized_perf)}, Episode Length: {ep_len}, Done: {done}')
        total_rewards += ep_rew
        total_lengths += ep_len
        total_normalized_perf.append(ep_normalized_perf)
        if config['eval_videos']:
            save_numpy_as_gif(np.array(frames), os.path.join(eval_video_path, 'ep_%d.gif' % ep))
    avg_normalized_perf = np.mean(total_normalized_perf)
    final_normalized_perf = np.mean(np.array(total_normalized_perf)[:, -1])
    avg_rewards = total_rewards / num_eval_eps
    avg_ep_length = total_lengths / num_eval_eps
    print(f'Final Performance (info_normalized_performance_final): {final_normalized_perf}')
    print(f'Average Performance (info_normalized_performance_mean): {avg_normalized_perf}')
    print(f'Average Rewards: {avg_rewards}')
    print(f'Average Episode Length: {avg_ep_length}')
