import numpy as np
import torch
import os
import wandb
import copy

from sb3 import utils
from sb3.sac_bc import SAC_BC

from envs.env import SoftGymEnvSB3

from softgym.utils.visualization import save_numpy_as_gif, make_grid
import matplotlib.pyplot as plt

from stable_baselines3 import SAC, TD3
from stable_baselines3.common.callbacks import CallbackList, BaseCallback
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env

from wandb.integration.sb3 import WandbCallback

class EvalCallback(BaseCallback):
    """
    For detail information regarding each method, please refer to https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html
    """
    def __init__(self, verbose=0, config=None, env_kwargs=None):
        super(EvalCallback, self).__init__(verbose)
        self.agent = config['agent']
        self.val_freq = config['val_freq']
        self.val_num_eps = config['val_num_eps']
        self.max_steps = config['max_steps']
        self.ckpt_saved_folder = config['ckpt_saved_folder']
        self.encoder_type = config['encoder_type']
        self.image_size = config['image_size']
        self.wandb_logging = config['wandb']
        tb_dir = config['tb_dir']
        self.video_dir = utils.make_dir(f'{tb_dir}/video')
        self.enable_normalize_obs = config['enable_normalize_obs']
        self.sb3_iterations = config['sb3_iterations']
        self.verbose = config['verbose']

    def _on_training_start(self):
        pass

    def _on_rollout_start(self):
        pass

    def _on_step(self):
        return True

    def _on_rollout_end(self):
        if self.num_timesteps % self.val_freq == 0:
            # normalize observations
            training_env = self.training_env.envs[0].env

            if self.enable_normalize_obs:
                if self.num_timesteps == self.sb3_iterations:
                    max_min_obs = {
                        'max_obs': self.training_env.envs[0].env.max_obs,
                        'min_obs': self.training_env.envs[0].env.min_obs
                    }
                    np.save('./data/max_min_obs.npy', max_min_obs)

            ckpt_file_path = f"{self.ckpt_saved_folder}model_{self.num_timesteps}"
            self.model.save(ckpt_file_path)
            if self.agent == 'sac':
                model_class = SAC
            elif self.agent == 'td3':
                model_class = TD3
            elif self.agent == 'sac-bc':
                model_class = SAC_BC
            model = model_class.load(ckpt_file_path, env=training_env)

            total_rewards, total_lengths, total_normalized_perf = 0, 0, []
            total_task_reward, total_il_reward = 0, 0
            all_frames = []
            for ep in range(self.val_num_eps):
                obs = training_env.reset(is_eval=True)
                ep_rew, ep_len, done, ep_normalized_perf = 0, 0, False, []
                frames = [training_env.get_image(128, 128)]
                while ep_len < self.max_steps and not done:
                    if self.encoder_type == 'pixel':
                        obs = utils.center_crop_image(obs, self.image_size)
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, done, info = training_env.step(action)
                    ep_len += 1
                    ep_rew += reward
                    ep_normalized_perf.append(info['normalized_performance'])
                    frames.append(training_env.get_image(128, 128))
                total_lengths += ep_len
                total_rewards += ep_rew
                total_normalized_perf.append(ep_normalized_perf)

                total_task_reward += training_env.ep_task_reward * training_env._scale_reward
                total_il_reward += training_env.ep_il_reward * training_env._scale_reward

                if len(all_frames) < 8:
                    all_frames.append(frames)

            avg_normalized_perf = np.mean(total_normalized_perf)
            final_normalized_perf = np.mean(np.array(total_normalized_perf)[:, -1])
            avg_rewards = total_rewards / self.val_num_eps
            avg_ep_length = total_lengths / self.val_num_eps
            avg_ep_task_reward = total_task_reward / self.val_num_eps
            avg_ep_il_reward = total_il_reward / self.val_num_eps

            # since wandb will read fron tensorboard too
            self.logger.record('val/info_normalized_performance_mean', avg_normalized_perf)
            self.logger.record('val/info_normalized_performance_final', final_normalized_perf)
            self.logger.record('val/avg_rews', avg_rewards)
            self.logger.record('val/avg_ep_length', avg_ep_length)
            self.logger.record('val/avg_ep_tasl_reward', avg_ep_task_reward)
            self.logger.record('val/avg_ep_il_reward', avg_ep_il_reward)
            self.logger.record('num_timesteps', self.num_timesteps)

            if self.verbose:
                # incase we need to print
                print(f'Average Performance (info_normalized_performance_mean): {avg_normalized_perf}')
                print(f'Average Rewards: {avg_rewards}')
                print(f'Average Episode Length: {avg_ep_length}')
            del model
            all_frames = np.array(all_frames).swapaxes(0, 1)
            all_frames = np.array([make_grid(np.array(frame), nrow=2, padding=3) for frame in all_frames])
            save_numpy_as_gif(all_frames, os.path.join(self.video_dir, '%d.gif' % self.num_timesteps))
        if self.agent == 'sac-bc':
            # HACK: need to update num_timesteps because sac-bc does not call env.step()
            # (it's used when collecting rollouts); therefore, self.num_timesteps will not
            # be updated. We need to manually update self.num_timesteps.
            self.num_timesteps += 1

    def _on_training_end(self):
        pass


class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self):
        # Log scalars

        # For offline RL, self.locals would not have 'env'
        if 'env' in self.locals:
            # _on_rollout_end is called after every bit of experience is collected, therefore,
            # check if the last time step is the last time step of the episode before logging
            env_list = [env.env for env in self.locals['env'].envs if env.time_step == env.horizon - 1]
            if env_list != []:
                reward_scale = self.locals['env'].envs[0].env._env._scale_reward # Assume all envs have same reward scale
                task_reward = np.mean([env.ep_task_reward for env in env_list]) * reward_scale
                il_reward = np.mean([env.ep_il_reward for env in env_list]) * reward_scale
                self.logger.record('rollout/ep_task_reward', task_reward)
                self.logger.record('rollout/ep_il_reward', il_reward)         
        return True


def run_task(args, env_kwargs):
    # single environment
    env = SoftGymEnvSB3(**env_kwargs)

    # Disable multi-environments due to worse performance
    # if args.debug:
    #     # single environment
    #     env = SoftGymEnvSB3(**env_kwargs)
    # else:
    #     # multi-processing environment
    #     env = make_vec_env(SoftGymEnvSB3, n_envs=args.num_envs, seed=args.seed, env_kwargs=env_kwargs, vec_env_cls=SubprocVecEnv)

    if args.agent == 'sac':
        # SAC without HER
        policy_kwargs = dict(activation_fn=torch.nn.Tanh, net_arch=dict(pi=[512, 512], qf=[512, 512]))
        model = SAC("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1, tensorboard_log=args.tb_dir, seed=args.seed, ent_coef=args.ent_coef)
    elif args.agent == 'sac-bc':
        policy_kwargs = dict(activation_fn=torch.nn.Tanh, net_arch=dict(pi=[512, 512], qf=[512, 512]))
        model = SAC_BC("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1, tensorboard_log=args.tb_dir, seed=args.seed)
    elif args.agent == 'td3':
        policy_kwargs = dict(activation_fn=torch.nn.Tanh, net_arch=dict(pi=[64, 64], qf=[64, 64]))
        model = TD3(
            "MlpPolicy",
            env,
            policy_kwargs=policy_kwargs,
            verbose=1,
            tensorboard_log=args.tb_dir,
            seed=args.seed,
            train_freq=1,
            batch_size=args.batch_size)
    else:
        raise NotImplementedError

    eval_callback = EvalCallback(config=args.__dict__, env_kwargs=env_kwargs)
    tb_callback = TensorboardCallback(verbose=0)
    if args.wandb:
        run = wandb.init(
            project="dmfd",
            config=args.__dict__,
            sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
            name=args.folder_name,
        )

        wandb_callback = WandbCallback(
            gradient_save_freq=100,
            verbose=2,
        )
        callbacks = CallbackList([wandb_callback, eval_callback, tb_callback])
    else:
        callbacks = CallbackList([eval_callback, tb_callback])

    if args.bc_model_ckpt_file:
        bc_pretrained_state_dict = torch.load(args.bc_model_ckpt_file)['state_dict']
        pretrained_dict = dict()
        for k, v in bc_pretrained_state_dict.items():
            if 'mu' in k or 'log_std' in k:
                # there is an extra '.0.' for mu and log_std layer from BC model
                k = k.replace('.0.', '.')
            pretrained_dict[k] = v
        model.actor.load_state_dict(pretrained_dict)

        # load critic weight
        # critic_state_dict = torch.load('data/sb3/RopeFlatten_SB3_sac_01.21.16.16/model_420000_policy.pth')
        # critic_dict = dict()
        # for k, v in critic_state_dict.items():
        #     if 'critic.' in k:
        #         k = k.replace('critic.', '')
        #         critic_dict[k] = v

        # critic_target_dict = dict()
        # for k, v in critic_state_dict.items():
        #     if 'critic_target.' in k:
        #         k = k.replace('critic_target.', '')
        #         critic_target_dict[k] = v

        # model.critic.load_state_dict(critic_dict)
        # model.critic_target.load_state_dict(critic_target_dict)

    if args.agent == 'sac-bc':
        # offline RL training
        utils.fill_replay_buffer(args.rsi_file, model)

    model.learn(total_timesteps=args.sb3_iterations, log_interval=1, tb_log_name=args.folder_name, callback=callbacks)

    if args.wandb:
        run.finish()
    print('Finished training...')
