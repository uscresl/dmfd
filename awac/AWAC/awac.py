import itertools
import numpy as np
import torch
from torch.optim import Adam
import time
from awac import core
from awac.utils.logx import EpochLogger
import torch.nn.functional as F
from envs.env import SoftGymEnvSB3
import copy
import wandb
from sb3 import utils
from softgym.utils.visualization import save_numpy_as_gif
import os
from torchvision import transforms

device = torch.device("cuda")
class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size=32, idxs=None):
        if idxs is None:
            idxs = np.random.randint(0, self.size, size=batch_size)

        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in batch.items()}


class ReplayBufferImageBased:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """
    def __init__(self, obs_dim, act_dim, size, enable_img_aug, enable_drq_loss, env_image_size):
        # images
        self.obs_img_buf = np.zeros(core.combined_shape(size, (3, env_image_size, env_image_size)), dtype=np.float32)
        self.obs2_img_buf = np.zeros(core.combined_shape(size, (3, env_image_size, env_image_size)), dtype=np.float32)
        # key_points
        self.obs_state_buf = np.zeros(core.combined_shape(size, obs_dim['key_point'].shape), dtype=np.float32)
        self.obs2_state_buf = np.zeros(core.combined_shape(size, obs_dim['key_point'].shape), dtype=np.float32)

        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

        self.aug_trans = torch.nn.Sequential(
            torch.nn.ReplicationPad2d(2),
            transforms.RandomCrop((env_image_size, env_image_size)))
        self.enable_img_aug = enable_img_aug
        self.enable_drq_loss = enable_drq_loss

    def store(self, obs, act, rew, next_obs, done):
        self.obs_img_buf[self.ptr] = obs['cam_rgb']
        self.obs2_img_buf[self.ptr] = next_obs['cam_rgb']

        self.obs_state_buf[self.ptr] = obs['key_point']
        self.obs2_state_buf[self.ptr] = next_obs['key_point']

        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size=32, idxs=None):
        if idxs is None:
            idxs = np.random.randint(0, self.size, size=batch_size)

        batch = dict(obs_img=self.obs_img_buf[idxs],
                     obs2_img=self.obs2_img_buf[idxs],
                     obs_state=self.obs_state_buf[idxs],
                     obs2_state=self.obs2_state_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])

        if self.enable_drq_loss:
            batch['obs_img_second'] = batch['obs_img'].copy()
            batch['obs2_img_second'] = batch['obs2_img'].copy()

        batch_tensor = {k: torch.as_tensor(v, dtype=torch.float32) for k, v in batch.items()}
        if self.enable_img_aug:
            batch_tensor['obs_img'] = self.aug_trans(batch_tensor['obs_img'])
            batch_tensor['obs2_img'] = self.aug_trans(batch_tensor['obs2_img'])

            if self.enable_drq_loss:
                batch_tensor['obs_img_second'] = self.aug_trans(batch_tensor['obs_img_second'])
                batch_tensor['obs2_img_second'] = self.aug_trans(batch_tensor['obs2_img_second'])

        return batch_tensor


class AWAC:

    def __init__(self, args, env_kwargs):
        """
        Soft Actor-Critic (SAC)


        Args:
            env_fn : A function which creates a copy of the environment.
                The environment must satisfy the OpenAI Gym API.

            actor_critic: The constructor method for a PyTorch Module with an ``act``
                method, a ``pi`` module, a ``q1`` module, and a ``q2`` module.
                The ``act`` method and ``pi`` module should accept batches of
                observations as inputs, and ``q1`` and ``q2`` should accept a batch
                of observations and a batch of actions as inputs. When called,
                ``act``, ``q1``, and ``q2`` should return:

                ===========  ================  ======================================
                Call         Output Shape      Description
                ===========  ================  ======================================
                ``act``      (batch, act_dim)  | Numpy array of actions for each
                                            | observation.
                ``q1``       (batch,)          | Tensor containing one current estimate
                                            | of Q* for the provided observations
                                            | and actions. (Critical: make sure to
                                            | flatten this!)
                ``q2``       (batch,)          | Tensor containing the other current
                                            | estimate of Q* for the provided observations
                                            | and actions. (Critical: make sure to
                                            | flatten this!)
                ===========  ================  ======================================

                Calling ``pi`` should return:

                ===========  ================  ======================================
                Symbol       Shape             Description
                ===========  ================  ======================================
                ``a``        (batch, act_dim)  | Tensor containing actions from policy
                                            | given observations.
                ``logp_pi``  (batch,)          | Tensor containing log probabilities of
                                            | actions in ``a``. Importantly: gradients
                                            | should be able to flow back into ``a``.
                ===========  ================  ======================================

            ac_kwargs (dict): Any kwargs appropriate for the ActorCritic object
                you provided to SAC.

            seed (int): Seed for random number generators.

            steps_per_epoch (int): Number of steps of interaction (state-action pairs)
                for the agent and the environment in each epoch.

            epochs (int): Number of epochs to run and train agent.

            replay_size (int): Maximum length of replay buffer.

            gamma (float): Discount factor. (Always between 0 and 1.)

            polyak (float): Interpolation factor in polyak averaging for target
                networks. Target networks are updated towards main networks
                according to:

                .. math:: \\theta_{\\text{targ}} \\leftarrow
                    \\rho \\theta_{\\text{targ}} + (1-\\rho) \\theta

                where :math:`\\rho` is polyak. (Always between 0 and 1, usually
                close to 1.)

            lr (float): Learning rate (used for both policy and value learning).

            alpha (float): Entropy regularization coefficient. (Equivalent to
                inverse of reward scale in the original SAC paper.)

            batch_size (int): Minibatch size for SGD.

            start_steps (int): Number of steps for uniform-random action selection,
                before running real policy. Helps exploration.

            update_after (int): Number of env interactions to collect before
                starting to do gradient descent updates. Ensures replay buffer
                is full enough for useful updates.

            update_every (int): Number of env interactions that should elapse
                between gradient descent updates. Note: Regardless of how long
                you wait between updates, the ratio of env steps to gradient steps
                is locked to 1.

            num_test_episodes (int): Number of episodes to test the deterministic
                policy at the end of each epoch.

            max_ep_len (int): Maximum length of trajectory / episode / rollout.

            logger_kwargs (dict): Keyword args for EpochLogger.

            save_freq (int): How often (in terms of gap between epochs) to save
                the current policy and value function.

        """
        ac_kwargs=dict()

        if args['is_eval']:
            self.logger = None
        else:
            logger_kwargs = {
                'output_dir': args.get('tb_dir', ''),
                'exp_name': args.get('folder_name', ''),
                'wandb_logging': args.get('wandb', False),
            }
            self.logger = EpochLogger(**logger_kwargs)
            # self.logger.save_config(locals()) # takes up a lot of storage

        if args['wandb']:
            self.wandb_run = wandb.init(
                project="cto-rl-manipulation",
                config=args,
                name=args['folder_name'],
            )
        else:
            self.wandb_run = None

        self.env = SoftGymEnvSB3(**env_kwargs)

        self.obs_dim = self.env.observation_space.shape
        self.act_dim = self.env.action_space.shape[0]

        # Action limit for clamping: critically, assumes all dimensions share the same bound!
        self.act_limit = self.env.action_space.high[0]

        env_image_size = args['env_kwargs']['env_image_size']

        # Create actor-critic module and target networks
        actor_critic = core.MLPActorCritic
        bc_model_ckpt_file = args.get('bc_model_ckpt_file', None)
        self.awac_policy = 'awac_img' if self.env.observation_mode == 'cam_rgb_key_point' else 'awac'
        self.ac = actor_critic(self.env.observation_space, self.env.action_space,
                               special_policy=self.awac_policy, bc_model_ckpt_file=bc_model_ckpt_file, env_image_size=env_image_size, **ac_kwargs)
        self.ac_targ = actor_critic(self.env.observation_space, self.env.action_space,
                                    special_policy=self.awac_policy, bc_model_ckpt_file=bc_model_ckpt_file, env_image_size=env_image_size, **ac_kwargs)
        self.ac_targ.load_state_dict(self.ac.state_dict())
        self.gamma = args.get('gamma', 0.99)

        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in self.ac_targ.parameters():
            p.requires_grad = False

        # List of parameters for both Q-networks (save this for convenience)
        self.q_params = itertools.chain(self.ac.q1.parameters(), self.ac.q2.parameters())

        # Experience buffer
        replay_size = args.get('awac_replay_size', int(2000000))
        enable_img_aug = args.get('enable_img_aug', False)
        self.enable_drq_loss = args.get('enable_drq_loss', False)
        if self.env.observation_mode == 'cam_rgb_key_point':
            self.replay_buffer = ReplayBufferImageBased(obs_dim=self.env.observation_space, act_dim=self.act_dim, size=replay_size, enable_img_aug=enable_img_aug, enable_drq_loss=self.enable_drq_loss, env_image_size=env_image_size)
        else:
            self.replay_buffer = ReplayBuffer(obs_dim=self.obs_dim, act_dim=self.act_dim,
                                            size=replay_size)

        # Count variables (protip: try to get a feel for how different size networks behave!)
        var_counts = tuple(
            core.count_vars(module) for module in [self.ac.pi, self.ac.q1, self.ac.q2])
        if self.logger:
            self.logger.log('\nNumber of parameters: \t pi: %d, \t q1: %d, \t q2: %d\n' % var_counts)
        self.algo = args.get('algo', 'AWAC')

        self.p_lr = args.get('p_lr', 3e-4)
        self.lr = args.get('lr', 3e-4)
        self.alpha = 0
        # # Algorithm specific hyperparams

        # Set up optimizers for policy and q-function
        self.pi_optimizer = Adam(self.ac.pi.parameters(), lr=self.p_lr, weight_decay=1e-4)
        self.q_optimizer = Adam(self.q_params, lr=self.lr)
        self.num_test_episodes = args.get('val_num_eps', 10)
        self.max_ep_len = args.get('max_ep_len', 75)
        self.epochs = args.get('epochs', 10000)
        self.steps_per_epoch = args.get('steps_per_epoch', 100)
        self.update_after = args.get('update_after', 0)
        self.update_every = args.get('update_every', 50)
        self.batch_size = args.get('batch_size', 1024)
        self.save_freq = args.get('save_freq', 1)
        self.polyak = args.get('polyak', 0.995)
        self.val_freq = args.get('val_freq', 10000)
        self.add_sac_loss = args.get('add_sac_loss', False)
        self.sac_loss_weight = args.get('sac_loss_weight', 0.0)

        if self.logger:
            # Set up model saving
            self.logger.setup_pytorch_saver(self.ac)
        print("Running Offline RL algorithm: {}".format(self.algo))

    def populate_replay_buffer(self, rsi_file):
        reference_states = np.load(rsi_file, allow_pickle=True)
        states = reference_states['ob_trajs']
        next_states = reference_states['ob_next_trajs']
        actions = reference_states['action_trajs']
        rewards = reference_states['reward_trajs']
        dones = reference_states['done_trajs']

        if self.env.observation_mode == 'cam_rgb_key_point':
            images = reference_states['ob_img_trajs']
            next_images = reference_states['ob_img_next_trajs']

        for ep_counter in range(states.shape[0]):
            for traj_counter in range(len(states[ep_counter])):
                if self.env.observation_mode == 'cam_rgb_key_point':
                    obs = {
                        'key_point': states[ep_counter][traj_counter],
                        'cam_rgb': images[ep_counter][traj_counter].transpose(2, 0, 1),
                    }
                    next_obs = {
                        'key_point': next_states[ep_counter][traj_counter],
                        'cam_rgb': next_images[ep_counter][traj_counter].transpose(2, 0, 1),
                    }
                    self.replay_buffer.store(
                        obs,
                        actions[ep_counter][traj_counter],
                        rewards[ep_counter][traj_counter],
                        next_obs,
                        dones[ep_counter][traj_counter],
                    )
                else:
                    self.replay_buffer.store(
                        states[ep_counter][traj_counter],
                        actions[ep_counter][traj_counter],
                        rewards[ep_counter][traj_counter],
                        next_states[ep_counter][traj_counter],
                        dones[ep_counter][traj_counter],
                    )
        print("Loaded dataset")


    # Set up function for computing SAC Q-losses
    def compute_loss_q(self, data):
        if self.env.observation_mode == 'cam_rgb_key_point':
            o, o_img, o2, o2_img = data['obs_state'], data['obs_img'], data['obs2_state'], data['obs2_img']
        else:
            o, o2 = data['obs'], data['obs2']
        a, r, d = data['act'].to(device), data['rew'].to(device), data['done'].to(device)

        if self.enable_drq_loss:
            o_img_second = data['obs_img_second']
            o2_img_second = data['obs2_img_second']

            with torch.no_grad():
                a2, logp_a2 = self.ac.pi(o2_img)
                # Target Q-values
                q1_pi_targ = self.ac_targ.q1(o2_img, o2, a2)
                q2_pi_targ = self.ac_targ.q2(o2_img, o2, a2)
                q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
                backup = r + self.gamma * (1 - d) * (q_pi_targ - self.alpha * logp_a2)

                a2_second, logp_a2_second = self.ac.pi(o2_img_second)
                # Target Q-values
                q1_pi_targ_second = self.ac_targ.q1(o2_img_second, o2, a2_second)
                q2_pi_targ_second = self.ac_targ.q2(o2_img_second, o2, a2_second)
                q_pi_targ_second = torch.min(q1_pi_targ_second, q2_pi_targ_second)
                backup_second = r + self.gamma * (1 - d) * (q_pi_targ_second - self.alpha * logp_a2_second)

                target_Q = (backup + backup_second) / 2

            # MSE loss against Bellman backup
            q1 = self.ac.q1(o_img, o, a)
            q2 = self.ac.q2(o_img, o, a)
            loss_q1 = ((q1 - target_Q) ** 2).mean()
            loss_q2 = ((q2 - target_Q) ** 2).mean()
            loss_q = loss_q1.cpu() + loss_q2.cpu()

            q1_second = self.ac.q1(o_img_second, o, a)
            q2_second = self.ac.q2(o_img_second, o, a)
            loss_q1_second = ((q1_second - target_Q) ** 2).mean()
            loss_q2_second = ((q2_second - target_Q) ** 2).mean()
            loss_q += loss_q1_second.cpu() + loss_q2_second.cpu()
        else:
            if self.env.observation_mode == 'cam_rgb_key_point':
                q1 = self.ac.q1(o_img, o, a)
                q2 = self.ac.q2(o_img, o, a)

                # Bellman backup for Q functions
                with torch.no_grad():
                    # Target actions come from *current* policy
                    a2, logp_a2 = self.ac.pi(o2_img)

                    # Target Q-values
                    q1_pi_targ = self.ac_targ.q1(o2_img, o2, a2)
                    q2_pi_targ = self.ac_targ.q2(o2_img, o2, a2)
                    q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
                    backup = r + self.gamma * (1 - d) * (q_pi_targ - self.alpha * logp_a2)

                # MSE loss against Bellman backup
                loss_q1 = ((q1 - backup) ** 2).mean()
                loss_q2 = ((q2 - backup) ** 2).mean()
                loss_q = loss_q1.cpu() + loss_q2.cpu()
            else:
                q1 = self.ac.q1(o, a)
                q2 = self.ac.q2(o, a)

                # Bellman backup for Q functions
                with torch.no_grad():
                    # Target actions come from *current* policy
                    a2, logp_a2 = self.ac.pi(o2)

                    # Target Q-values
                    q1_pi_targ = self.ac_targ.q1(o2, a2)
                    q2_pi_targ = self.ac_targ.q2(o2, a2)
                    q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
                    backup = r + self.gamma * (1 - d) * (q_pi_targ - self.alpha * logp_a2)

                # MSE loss against Bellman backup
                loss_q1 = ((q1 - backup) ** 2).mean()
                loss_q2 = ((q2 - backup) ** 2).mean()
                loss_q = loss_q1.cpu() + loss_q2.cpu()

        # Useful info for logging
        q_info = dict(Q1Vals=q1.cpu().detach().numpy(),
                      Q2Vals=q2.cpu().detach().numpy())

        return loss_q, q_info

    # Set up function for computing SAC pi loss
    def compute_loss_pi(self, data):
        if self.env.observation_mode == 'cam_rgb_key_point':
            o, o_img = data['obs_state'], data['obs_img']
            pi, logp_pi = self.ac.pi(o_img)
        else:
            o = data['obs']
            pi, logp_pi = self.ac.pi(o)
        replay_buf_act = data['act'].to(device)

        if self.env.observation_mode == 'cam_rgb_key_point':
            q1_pi = self.ac.q1(o_img, o, pi)
            q2_pi = self.ac.q2(o_img, o, pi)
        else:
            q1_pi = self.ac.q1(o, pi)
            q2_pi = self.ac.q2(o, pi)
        v_pi = torch.min(q1_pi, q2_pi)

        beta = 2
        if self.env.observation_mode == 'cam_rgb_key_point':
            q1_old_actions = self.ac.q1(o_img, o, replay_buf_act)
            q2_old_actions = self.ac.q2(o_img, o, replay_buf_act)
        else:
            q1_old_actions = self.ac.q1(o, replay_buf_act)
            q2_old_actions = self.ac.q2(o, replay_buf_act)
        q_old_actions = torch.min(q1_old_actions, q2_old_actions)

        adv_pi = q_old_actions - v_pi
        weights = F.softmax(adv_pi / beta, dim=0)
        if self.env.observation_mode == 'cam_rgb_key_point':
            policy_logpp = self.ac.pi.get_logprob(o_img, replay_buf_act)
        else:
            policy_logpp = self.ac.pi.get_logprob(o, replay_buf_act)
        awac_loss_pi = (-policy_logpp * len(weights) * weights.detach()).mean()

        if self.add_sac_loss:
            # SAC actor loss + AWAC actor loss
            ent_coef = 0.5
            sac_loss_pi = (ent_coef * logp_pi - v_pi).mean()
            loss_pi = ((1 - self.sac_loss_weight) * awac_loss_pi) + (self.sac_loss_weight * sac_loss_pi)
        else:
            # just AWAC actor loss
            loss_pi = awac_loss_pi

        # Useful info for logging
        pi_info = dict(LogPi=policy_logpp.cpu().detach().numpy())

        return loss_pi, pi_info

    def update(self, data, update_timestep):
        # First run one gradient descent step for Q1 and Q2
        self.q_optimizer.zero_grad()
        loss_q, q_info = self.compute_loss_q(data)
        loss_q.backward()
        self.q_optimizer.step()

        # Record things
        self.logger.store(LossQ=loss_q.item(), **q_info)
        # Freeze Q-networks so you don't waste computational effort
        # computing gradients for them during the policy learning step.
        for p in self.q_params:
            p.requires_grad = False

        # Next run one gradient descent step for pi.
        self.pi_optimizer.zero_grad()
        loss_pi, pi_info = self.compute_loss_pi(data)
        loss_pi.backward()
        self.pi_optimizer.step()

        # Unfreeze Q-networks so you can optimize it at next DDPG step.
        for p in self.q_params:
            p.requires_grad = True

        # Record things
        self.logger.store(LossPi=loss_pi.item(), **pi_info)

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(self.ac.parameters(), self.ac_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)

    def get_action(self, o, deterministic=False):
        return self.ac.act(torch.as_tensor(o, dtype=torch.float32), deterministic)

    def get_action_image_based(self, o, deterministic=False):
        return self.ac.act(o, deterministic)

    def eval_agent_five_seeds(self, config):
        """
        Evaluation script over five seeds.
        Output the mean, std, median, 25th, and 75 percentiles.
        """
        self.ac.load_state_dict(torch.load(config['checkpoint']).state_dict())
        self.ac.eval()

        total_normalized_perf_final = []
        random_seeds = [100, 201, 302, 403, 504]
        for curr_seed in random_seeds:
            utils.set_seed_everywhere(curr_seed)
            for ep in range(20):
                obs = self.env.reset(is_eval=True)
                done, ep_len, ep_rew, ep_normalized_perf = False, 0, 0, []
                while ep_len < config['max_steps'] and not done:
                    if self.env.observation_mode == 'cam_rgb_key_point':
                        action = self.get_action_image_based(obs, True)
                    else:
                        action = self.get_action(obs, True)
                    obs, reward, done, info = self.env.step(action)
                    ep_len += 1
                    ep_rew += reward
                    ep_normalized_perf.append(info['normalized_performance'])
                print(f'Seed {curr_seed} Ep {ep} Current Episode Rewards: {ep_rew}, Episode normalized performance final: {ep_normalized_perf[-1]}, Episode Length: {ep_len}, Done: {done}')
                total_normalized_perf_final.append(ep_normalized_perf[-1])

        total_normalized_perf_final = np.array(total_normalized_perf_final)

        ckpt_file_path = config['checkpoint']
        npy_file_path = "/".join(ckpt_file_path.split('/')[:-1]) + '/' + ckpt_file_path.split('-ckpt-')[0].split('/')[-1] + '.npy'
        np.save(npy_file_path, total_normalized_perf_final)
        print('!!!!!!! info_normalized_performance_final !!!!!!!')
        print(f'Mean: {np.mean(total_normalized_perf_final):.4f}')
        print(f'Std: {np.std(total_normalized_perf_final):.4f}')
        print(f'Median: {np.median(total_normalized_perf_final):.4f}')
        print(f'25th Percentile: {np.percentile(total_normalized_perf_final, 25):.4f}')
        print(f'75th Percentile: {np.percentile(total_normalized_perf_final, 75):.4f}')

    def eval_agent(self, config):
        """
        Evaluation script
        """
        if config['eval_videos']:
            eval_video_path = utils.make_dir('/'.join(config['checkpoint'].split('/')[:-2]) + '/eval_video')

        self.ac.load_state_dict(torch.load(config['checkpoint']).state_dict())
        self.ac.eval()

        total_rewards, total_lengths, total_normalized_perf = 0, 0, []
        num_eval_eps = config['num_eval_eps']
        for ep in range(num_eval_eps):
            obs = self.env.reset(is_eval=True)
            done, ep_len, ep_rew, ep_normalized_perf = False, 0, 0, []
            if config['eval_videos']:
                frames = [self.env.get_image(config['eval_gif_size'], config['eval_gif_size'])]
            while ep_len < config['max_steps'] and not done:
                if self.env.observation_mode == 'cam_rgb_key_point':
                    action = self.get_action_image_based(obs, True)
                else:
                    action = self.get_action(obs, True)
                obs, reward, done, info = self.env.step(action)
                ep_len += 1
                ep_rew += reward
                ep_normalized_perf.append(info['normalized_performance'])
                if config['eval_videos']:
                    frames.append(self.env.get_image(config['eval_gif_size'], config['eval_gif_size']))
            print(f'Ep {ep} Current Episode Rewards: {ep_rew}, Episode normalized performance final: {ep_normalized_perf[-1]}, Episode Length: {ep_len}, Done: {done}')
            total_rewards += ep_rew
            total_lengths += ep_len
            total_normalized_perf.append(ep_normalized_perf)
            if config['eval_videos']:
                save_numpy_as_gif(np.array(frames), os.path.join(eval_video_path, f'ep_{ep}_{ep_normalized_perf[-1]}.gif'))
        avg_normalized_perf = np.mean(total_normalized_perf)
        final_normalized_perf = np.mean(np.array(total_normalized_perf)[:, -1])
        avg_rewards = total_rewards / num_eval_eps
        avg_ep_length = total_lengths / num_eval_eps
        print(f'Final Performance (info_normalized_performance_final): {final_normalized_perf}')
        print(f'Average Performance (info_normalized_performance_mean): {avg_normalized_perf}')
        print(f'Average Rewards: {avg_rewards}')
        print(f'Average Episode Length: {avg_ep_length}')

    def test_agent(self, ckpt_path, t):
        """
        Validation script during training
        """
        # create a new evaluation actor with saved checkpoint
        ac_kwargs = dict()
        test_ac = core.MLPActorCritic(self.env.observation_space, self.env.action_space,
                               special_policy=self.awac_policy, bc_model_ckpt_file=None, **ac_kwargs)
        test_ac.load_state_dict(torch.load(ckpt_path).state_dict())
        test_ac.eval()

        total_rewards, total_lengths, total_normalized_perf = 0, 0, []
        for j in range(self.num_test_episodes):
            o, d, ep_ret, ep_len = self.env.reset(is_eval=True), False, 0, 0
            ep_normalized_perf = []
            while not (d or (ep_len == self.max_ep_len)):
                # Take deterministic actions at test time
                if self.env.observation_mode == 'cam_rgb_key_point':
                    policy_act = test_ac.act(o, True)
                else:
                    policy_act = test_ac.act(torch.as_tensor(o, dtype=torch.float32), True)
                o, r, d, info = self.env.step(policy_act)
                ep_ret += r
                ep_len += 1
                ep_normalized_perf.append(info['normalized_performance'])

            total_rewards += ep_ret
            total_lengths += ep_len
            total_normalized_perf.append(ep_normalized_perf)

        avg_rewards = total_rewards / self.num_test_episodes
        avg_ep_length = total_lengths / self.num_test_episodes
        avg_normalized_perf = np.mean(total_normalized_perf)
        final_normalized_perf = np.mean(np.array(total_normalized_perf)[:, -1])
        if self.wandb_run:
            wandb.log({
                "val/info_normalized_performance_mean": avg_normalized_perf,
                'val/info_normalized_performance_final': final_normalized_perf,
                "val/avg_rews": avg_rewards,
                "val/avg_ep_length": avg_ep_length,
                "num_timesteps": t,
            })

    def run(self, args):
        # Prepare for interaction with environment
        total_steps = args.sb3_iterations + 1
        # total_steps = self.epochs * self.steps_per_epoch
        start_time = time.time()
        obs, ep_ret, ep_len = self.env.reset(), 0, 0
        done = True

        # Main loop: collect experience in env and update/log each epoch
        for t in range(total_steps):

            # Reset stuff if necessary
            if done and t > 0:
                self.logger.store(ExplEpRet=ep_ret, ExplEpLen=ep_len)

                obs, ep_ret, ep_len = self.env.reset(), 0, 0

            # Collect experience
            if self.env.observation_mode == 'cam_rgb_key_point':
                act = self.get_action_image_based(obs, deterministic=False)
            else:
                act = self.get_action(obs, deterministic=False)
            next_obs, rew, done, info = self.env.step(act)

            self.replay_buffer.store(obs, act, rew, next_obs, done)
            obs = next_obs

            # Update handling
            if t > self.update_after and t % self.update_every == 0:
                for _ in range(self.update_every):
                    batch = self.replay_buffer.sample_batch(self.batch_size)
                    self.update(data=batch, update_timestep=t)

            # End of epoch handling
            if (t + 1) % self.steps_per_epoch == 0:
                epoch = (t + 1) // self.steps_per_epoch

                # Log info about epoch
                self.logger.log_tabular('Epoch', epoch)
                self.logger.log_tabular('num_timesteps', t)
                self.logger.log_tabular('Q1Vals', with_min_and_max=True)
                self.logger.log_tabular('Q2Vals', with_min_and_max=True)
                self.logger.log_tabular('LogPi', with_min_and_max=True)
                self.logger.log_tabular('LossPi', average_only=True)
                self.logger.log_tabular('LossQ', average_only=True)
                self.logger.log_tabular('Time', time.time() - start_time)
                self.logger.dump_tabular()

            if t != 0 and t % self.val_freq == 0:
                # Save model
                ckpt_path = self.logger.save_state({'env': self.env}, itr=t)

                # Test the performance of the deterministic version of the agent.
                self.test_agent(ckpt_path, t)

        if self.wandb_run:
            self.wandb_run.finish()
