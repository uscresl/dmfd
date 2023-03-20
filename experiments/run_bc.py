import tqdm
import wandb
import torch.optim as optim
from softgym.registered_env import env_arg_dict
from envs.env import SoftGymEnvSB3
import argparse
import os
import pickle

import numpy as np
import torch
import torch.nn as nn
from sb3.utils import str2bool, set_seed_everywhere, update_env_kwargs, make_dir
from softgym.utils.visualization import save_numpy_as_gif
from torch.utils.data import DataLoader, Dataset
from datetime import datetime
from torch.distributions import Normal
import math
from torchvision import transforms

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

parser = argparse.ArgumentParser()

## general arguments
parser.add_argument('--is_eval', type=str2bool, default=False, help="evaluation or training mode")
parser.add_argument('--is_image_based', type=str2bool, default=False, help="state-based or image-based observations")
parser.add_argument('--is_bc_for_sac', type=str2bool, default=False, help="whether or not we're training a BC-Visual for SAC")
parser.add_argument('--enable_img_transformations', type=str2bool, default=False, help="Whether to enable image transformations")
parser.add_argument('--load_ob_image_mode', default='separate_folder', choices=['direct', 'separate_folder'], help='direct: load all images in memory; separate_folder: only load mini-batch images in memory')
parser.add_argument('--eval_videos', type=str2bool, default=False, help="whether or not to save evaluation video per episode")
parser.add_argument('--eval_gif_size',  default=512, type=int, help="evaluation GIF width and height size")
parser.add_argument('--model_save_dir', type=str, default='./data/bc', help="directory for saving trained model weights")
parser.add_argument('--saved_rollouts', type=str, default=None, help="directory to load saved expert demonstrations from")
parser.add_argument('--seed', type=int, default=1234, help="torch seed value")

## training arguments
parser.add_argument('--train_data_ratio', type=float, default=0.90, help="ratio for training data for train-test split")
parser.add_argument('--max_train_epochs', type=int, default=10000, help="ending epoch for training")

## validation arguments
parser.add_argument('--eval_interval', type=int, default=10, help="evaluation_interval")

## test arguments
parser.add_argument('--test_checkpoint', type=str, default='./checkpoints/epoch_0.pth', help="checkpoint file for evaluation")
parser.add_argument('--eval_over_five_seeds', default=False, type=str2bool, help="evaluation over 5 random seeds (100 episodes per seed)")

## arguments used in both validation and test
parser.add_argument('--num_eval_eps', type=int, default=50, help="number of episodes to run during evaluation")

## logs
parser.add_argument('--wandb', action='store_true', help="learning curves logged on weights and biases")
parser.add_argument('--name', default=None, type=str, help='[optional] set experiment name. Useful to resume experiments.')

## model arguments
parser.add_argument('--lrate', type=float, default=0.0005, help="initial learning rate for the policy network update")
parser.add_argument('--beta1', type=float, default=0.95, help="betas for Adam Optimizer")
parser.add_argument('--beta2', type=float, default=0.9, help="betas for Adam Optimizer")
parser.add_argument('--batch_size', type=int, default=512, help="batch size for model training")
parser.add_argument('--scheduler_step_size', type=int, default=5, help="step size for optimizer scheduler")
parser.add_argument('--scheduler_gamma', type=float, default=0.99, help="decay rate for optimizer scheduler")
parser.add_argument('--discount_factor', type=float, default=0.99, help="discount factor for calculating discounted rewards")
parser.add_argument('--observation_size', type=int, default=36, help="dimension of the action space")
parser.add_argument('--action_size', type=int, default=8, help="dimension of the action space")

## environment arguments
parser.add_argument('--env_name', default='RopeFlatten')
parser.add_argument('--env_img_size', type=int, default=128, help='Environment (observation) image size')
parser.add_argument('--env_kwargs_render', default=True, type=bool)  # Turn off rendering can speed up training
parser.add_argument('--env_kwargs_camera_name', default='default_camera', type=str)
parser.add_argument('--env_kwargs_observation_mode', default='key_point', type=str)  # Should be in ['key_point', 'cam_rgb', 'point_cloud']
parser.add_argument('--env_kwargs_num_variations', default=1000, type=int)

args = parser.parse_args()

# set env_specific parameters
env_name = args.env_name
obs_mode = args.env_kwargs_observation_mode
args.scale_reward = reward_scales[env_name]
args.clip_obs = clip_obs[env_name] if obs_mode == 'key_point' else None
args.env_kwargs = env_arg_dict[env_name]
args.__dict__ = update_env_kwargs(args.__dict__)  # Update env_kwargs

symbolic = args.env_kwargs['observation_mode'] != 'cam_rgb'
args.encoder_type = 'identity' if symbolic else 'pixel'
args.max_steps = 200
env_kwargs = {
    'env': args.env_name,
    'symbolic': symbolic,
    'seed': args.seed,
    'max_episode_length': args.max_steps,
    'action_repeat': 1,
    'bit_depth': 8,
    'image_dim': None,
    'env_kwargs': args.env_kwargs,
    'normalize_observation': False,
    'scale_reward': args.scale_reward,
    'clip_obs': args.clip_obs,
    'obs_process': None,
}
now = datetime.now().strftime("%m.%d.%H.%M")
args.folder_name = f'{args.env_name}_BC_{now}' if not args.name else args.name

# fix random seed
set_seed_everywhere(args.seed)

class BC(nn.Module):
    def __init__(self, obs_size=0, action_size=2):
        super(BC, self).__init__()

        self.latent_pi = nn.Sequential(
            nn.Linear(obs_size, 512),
            nn.Tanh(),
            nn.Linear(512, 512),
            nn.Tanh(),
        )

        self.mu = nn.Sequential(
            nn.Linear(512, action_size),
        )

        self.log_std = nn.Sequential(
            nn.Linear(512, action_size),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        references:
        https://github.com/DLR-RM/stable-baselines3/blob/54bcfa4544315fc920be0944fc380fd75e2f7c4a/stable_baselines3/common/distributions.py
        https://github.com/clvrai/mopa-pd/blob/master/rl/behavioral_cloning_visual.py#L160
        """
        latent_pi = self.latent_pi(x)
        means = self.mu(latent_pi)
        log_std = self.log_std(latent_pi)
        log_std = torch.clamp(log_std, -20, 2) # LOG_STD_MIN, LOG_STD_MAX from https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/sac/policies.py
        stds = torch.exp(log_std)
        z = Normal(means, stds).rsample()
        action = torch.tanh(z)
        return action

class BC_Visual(nn.Module):
    def __init__(self, action_size=2):
        super(BC_Visual, self).__init__()

        img_size = args.env_img_size
        first_linear_layer_size = int(256 * math.floor(img_size / 8) * math.floor(img_size / 8))

        self.cnn_layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
			nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

        self.latent_pi = nn.Sequential(
            nn.Linear(first_linear_layer_size, 512),
            nn.Tanh(),
            nn.Linear(512, 512),
            nn.Tanh(),
        )

        self.mu = nn.Sequential(
            nn.Linear(512, action_size),
        )

        self.log_std = nn.Sequential(
            nn.Linear(512, action_size),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        references:
        https://github.com/DLR-RM/stable-baselines3/blob/54bcfa4544315fc920be0944fc380fd75e2f7c4a/stable_baselines3/common/distributions.py
        https://github.com/clvrai/mopa-pd/blob/master/rl/behavioral_cloning_visual.py#L160
        """
        x = self.cnn_layers(x)
        x = torch.flatten(x, 1)
        latent_pi = self.latent_pi(x)
        means = self.mu(latent_pi)
        log_std = self.log_std(latent_pi)
        log_std = torch.clamp(log_std, -20, 2) # LOG_STD_MIN, LOG_STD_MAX from https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/sac/policies.py
        stds = torch.exp(log_std)
        z = Normal(means, stds).rsample()
        action = torch.tanh(z)
        return action

class BC_Visual_SAC_Actor(nn.Module):
    def __init__(self, action_size=2):
        super(BC_Visual_SAC_Actor, self).__init__()

        img_size = args.env_img_size

        self.cnn_layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1),
            nn.LeakyReLU(),
			nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1),
            nn.LeakyReLU(),
        )

        test_mat = torch.zeros(1, 3, img_size, img_size) # (1, img_channel, env_image_size, env_image_size)
        for conv_layer in self.cnn_layers:
            test_mat = conv_layer(test_mat)
        fc_input_size = int(np.prod(test_mat.shape))

        self.head = nn.Sequential(
            nn.Linear(fc_input_size, 50),
            nn.LayerNorm(50))

        self.trunk = nn.Sequential(
            nn.Linear(50, 1024), nn.ReLU(),
            nn.Linear(1024, 1024), nn.ReLU(),
            nn.Linear(1024, 2 * action_size)
        )

        self.log_std_min = -10
        self.log_std_max = 2

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        CoLActor in curl_sac.py
        """
        x = self.cnn_layers(x)
        x = torch.flatten(x, 1)
        latent_pi = self.head(x)
        mu, log_std = self.trunk(latent_pi).chunk(2, dim=-1)

        log_std = torch.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (
          self.log_std_max - self.log_std_min
        ) * (log_std + 1)

        std = log_std.exp()
        noise = torch.randn_like(mu)
        pi = mu + noise * std

        return pi

class Demonstrations(Dataset):
    def __init__(self, file_path):
        self.data = self.load_file(file_path)
        self.transform_normalize = transforms.Compose([
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.transform_randoms = transforms.RandomApply(torch.nn.ModuleList([
                                            transforms.RandomHorizontalFlip(p=1.0),
                                            transforms.RandomVerticalFlip(p=1.0),
                                            ]), p=0.5)
    def __len__(self):
        return len(self.data['obs'])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        ob, act = self.data["obs"][idx], self.data["actions"][idx]
        if args.is_image_based and args.load_ob_image_mode == 'separate_folder':
            ob = np.load(ob)
            ob = np.transpose(ob, (2, 0, 1))

        if args.is_image_based and args.enable_img_transformations:
            ob = self.transform_normalize(np.transpose(ob, (1, 2, 0)))
            # ob = self.transform_randoms(ob) # seems to perform worse based on one experiment
        else:
            ob = torch.from_numpy(ob)
        act = torch.from_numpy(act)

        out = {'ob': ob, 'act': act}
        return out

    def load_file(self, file_path):
        print('loading all data to RAM before training....')
        final_data = {
            'obs': [],
            'actions': [],
        }
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            if args.is_image_based:
                if args.load_ob_image_mode == 'direct':
                    ob_trajs = data['ob_img_trajs']
                    # original (num_episodes, 75, 128, 128, 3) to (num_episodes, 75, 3, 128, 128)
                    ob_trajs = np.transpose(ob_trajs, (0, 1, 4, 2, 3))
                elif args.load_ob_image_mode == 'separate_folder':
                    ob_trajs = data['ob_img_trajs']
            else:
                if 'ob_trajs' in data:
                    ob_trajs = data['ob_trajs']
                elif 'obs_trajs' in data:
                    # for handling v1 file (inconsistent naming against cem_trajs)
                    ob_trajs = data['obs_trajs']
            action_trajs = data['action_trajs']

            for obs_ep, act_ep in zip(ob_trajs, action_trajs):
                for ob, act in zip(obs_ep, act_ep):
                    final_data['obs'].append(ob)
                    final_data['actions'].append(act)
            final_data['obs'] = np.array(final_data['obs'])
            final_data['actions'] = np.array(final_data['actions'])
        print('finished loading data.')
        return final_data


class Evaluation:
    def __init__(self):
        self.env = SoftGymEnvSB3(**env_kwargs)
        if args.eval_videos:
            self.eval_video_path = make_dir('/'.join(args.test_checkpoint.split('/')[:-1]) + '/eval_video')


    def evaluate(self, checkpoint):
        if args.is_image_based:
            if args.is_bc_for_sac:
                model_eval = BC_Visual_SAC_Actor(action_size=args.action_size)
            else:
                model_eval = BC_Visual(action_size=args.action_size)
        else:
            model_eval = BC(obs_size=args.observation_size, action_size=args.action_size)

        model_eval.cuda()
        model_eval.load_state_dict(checkpoint['state_dict'])
        model_eval.eval()

        total_normalized_performance_final, total_rewards, total_lengths = [], 0, 0
        for ep in range(args.num_eval_eps):
            self.env.reset()
            if args.is_image_based:
                obs = self.env.get_image(args.env_img_size, args.env_img_size).transpose(2, 0, 1)
                obs = torch.from_numpy(np.expand_dims(obs, axis=0)).float().cuda()
            else:
                obs = torch.from_numpy(self.env._get_obs()).float().cuda()

            ep_len = 0
            ep_rew = 0
            ep_normalized_perf = []
            if args.eval_videos:
                frames = [self.env.get_image(args.eval_gif_size, args.eval_gif_size)]
            while ep_len < self.env.horizon:
                ac_pred = model_eval(obs)
                obs, rew, _, info = self.env.step(ac_pred.cpu().detach().numpy())
                if args.is_image_based:
                    obs = self.env.get_image(args.env_img_size, args.env_img_size).transpose(2, 0, 1)
                    obs = torch.from_numpy(np.expand_dims(obs, axis=0)).float().cuda()
                else:
                    obs = torch.from_numpy(obs).float().cuda()
                ep_len += 1
                ep_rew += rew
                ep_normalized_perf.append(info['normalized_performance'])
                if args.eval_videos:
                    frames.append(self.env.get_image(args.eval_gif_size, args.eval_gif_size))
            ep_normalized_perf_final = ep_normalized_perf[-1]
            print(f'Episode {ep}, Episode normalized performance final: {ep_normalized_perf_final}, Rewards: {ep_rew}, Episode Length: {ep_len}')
            total_normalized_performance_final.append(ep_normalized_perf_final)
            total_rewards += ep_rew
            total_lengths += ep_len
            if args.eval_videos:
                save_numpy_as_gif(np.array(frames), os.path.join(self.eval_video_path, f'ep_{ep}_{ep_normalized_perf_final}.gif'))
        del model_eval
        normalized_performance_final = np.mean(total_normalized_performance_final)
        avg_rewards = total_rewards / args.num_eval_eps
        avg_ep_length = total_lengths / args.num_eval_eps
        print(f'info_normalized_performance_final: {normalized_performance_final}')
        print(f'Average Rewards: {avg_rewards}')
        print(f'Average Episode Length: {avg_ep_length}')
        return normalized_performance_final, avg_rewards, avg_ep_length

    def evaluate_five_seeds(self, checkpoint):
        if args.is_image_based:
            if args.is_bc_for_sac:
                model_eval = BC_Visual_SAC_Actor(action_size=args.action_size)
            else:
                model_eval = BC_Visual(action_size=args.action_size)
        else:
            model_eval = BC(obs_size=args.observation_size, action_size=args.action_size)

        model_eval.cuda()
        model_eval.load_state_dict(checkpoint['state_dict'])
        model_eval.eval()

        total_normalized_perf_final = []
        random_seeds = [100, 201, 302, 403, 504]
        for curr_seed in random_seeds:
            set_seed_everywhere(curr_seed)
            for ep in range(20):
                self.env.reset()
                if args.is_image_based:
                    obs = self.env.get_image(args.env_img_size, args.env_img_size).transpose(2, 0, 1)
                    obs = torch.from_numpy(np.expand_dims(obs, axis=0)).float().cuda()
                else:
                    obs = torch.from_numpy(self.env._get_obs()).float().cuda()

                done, ep_normalized_perf = False, []
                while not done:
                    ac_pred = model_eval(obs)
                    obs, rew, done, info = self.env.step(ac_pred.cpu().detach().numpy())
                    if args.is_image_based:
                        obs = self.env.get_image(args.env_img_size, args.env_img_size).transpose(2, 0, 1)
                        obs = torch.from_numpy(np.expand_dims(obs, axis=0)).float().cuda()
                    else:
                        obs = torch.from_numpy(obs).float().cuda()
                    ep_normalized_perf.append(info['normalized_performance'])
                ep_normalized_perf_final = ep_normalized_perf[-1]
                print(f'Seed {curr_seed} Episode {ep}, Episode normalized performance final: {ep_normalized_perf_final}')
                total_normalized_perf_final.append(ep_normalized_perf_final)

        total_normalized_perf_final = np.array(total_normalized_perf_final)

        ckpt_file_path = args.test_checkpoint
        npy_file_path = "/".join(ckpt_file_path.split('/')[:-1]) + '/' + ckpt_file_path.split('-ckpt-')[0].split('/')[-1] + '.npy'
        np.save(npy_file_path, total_normalized_perf_final)

        print('!!!!!!! info_normalized_performance_final !!!!!!!')
        print(f'Mean: {np.mean(total_normalized_perf_final):.4f}')
        print(f'Std: {np.std(total_normalized_perf_final):.4f}')
        print(f'Median: {np.median(total_normalized_perf_final):.4f}')
        print(f'25th Percentile: {np.percentile(total_normalized_perf_final, 25):.4f}')
        print(f'75th Percentile: {np.percentile(total_normalized_perf_final, 75):.4f}')

def main_training():
    # get device
    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.cuda.empty_cache()
    else:
        device = torch.device('cpu')
    print(f"Device set to : {device}")

    mse_loss = nn.MSELoss()
    if args.is_image_based:
        if args.is_bc_for_sac:
            model = BC_Visual_SAC_Actor(action_size=args.action_size)
        else:
            model = BC_Visual(action_size=args.action_size)
    else:
        model = BC(obs_size=args.observation_size, action_size=args.action_size)
    optimizer = optim.Adam(list(model.parameters()), lr = args.lrate, betas = (args.beta1, args.beta2))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_step_size, gamma=args.scheduler_gamma)

    if args.wandb:
        wandb.init(
            project="dmfd",
            config={k: v for k, v in args.__dict__.items()},
            name=args.folder_name,
        )
        wandb.watch(model)
        model_save_dir = os.path.join(args.model_save_dir, wandb.run.name)
    else:
        model_save_dir = os.path.join(args.model_save_dir, 'tmp')
    print(f'Training model:\n{model}')


    dataset = Demonstrations(args.saved_rollouts)
    dataset_length = len(dataset)
    train_size = int(args.train_data_ratio * dataset_length)
    test_size = dataset_length - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    dataloader_train = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, drop_last=True)
    dataloader_val = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)

    model.cuda()
    mse_loss.cuda()

    evaluation_obj = Evaluation()

    print('Total number of state-action pairs: ', dataset_length)
    print('Number of training state-action pairs: ', len(train_dataset))
    print('Number of validation state-action pairs: ', len(val_dataset))

    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)

    outer = tqdm.tqdm(total=args.max_train_epochs, desc='Epoch', position=0)
    for epoch in range(args.max_train_epochs):
        total_loss = 0.0
        validation_loss = 0.0

        model.train()

        print('\nprocessing training batch...')
        for i, batch in enumerate(dataloader_train):
            ob, act = batch['ob'], batch['act']
            ob = ob.float().cuda()
            act = act.float().cuda()

            ac_pred = model(ob)

            ac_predictor_loss = mse_loss(ac_pred, act)
            optimizer.zero_grad()
            ac_predictor_loss.backward()
            optimizer.step()
            total_loss += ac_predictor_loss.data.item()

        scheduler.step()
        training_loss = total_loss / (args.batch_size*len(dataloader_train))
        print('\n----------------------------------------------------------------------')
        print('Epoch #' + str(epoch))
        print('Action Prediction Loss (Train): ' + str(training_loss))
        print('----------------------------------------------------------------------')

        # evaluating on test set
        model.eval()

        action_predictor_loss_val = 0.

        print('\nprocessing validation batch...')
        for i, batch in enumerate(dataloader_val):
            ob, act = batch['ob'], batch['act']
            ob = ob.float().cuda()
            act = act.float().cuda()

            ac_pred = model(ob)

            action_predictor_loss_val = mse_loss(ac_pred, act)
            validation_loss += action_predictor_loss_val.data.item()

        validation_loss /= (args.batch_size * len(dataloader_val))

        print('\n**********************************************************************')
        print('Epoch #' + str(epoch))
        print('')
        print('Action Prediction Loss (Validation): ' + str(validation_loss))
        print()
        print('**********************************************************************')

        # arrange/save checkpoint
        checkpoint = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
        }
        torch.save(checkpoint, os.path.join(model_save_dir, 'epoch_{}.pth'.format(epoch)))

        # perform validation
        if epoch % args.eval_interval == 0:
            normalized_performance_final, avg_rewards, avg_ep_length = evaluation_obj.evaluate(checkpoint)
        else:
            normalized_performance_final, avg_rewards, avg_ep_length = -1, -1, -1

        # wandb logging
        if args.wandb:
            if epoch % args.eval_interval == 0:
                wandb.log({
                    "Epoch": epoch,
                    "val/info_normalized_performance_final": normalized_performance_final,
                    "val/avg_rews": avg_rewards,
                    "val/avg_ep_length": avg_ep_length,
                    "Action Prediction Loss (Train)": training_loss,
                    "Action Prediction Loss (Validation)": validation_loss,
                })
            else:
                wandb.log({
                    "Epoch": epoch,
                    "Action Prediction Loss (Train)": training_loss,
                    "Action Prediction Loss (Validation)": validation_loss,
                })
        outer.update(1)

def main_testing():
    # get device
    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.cuda.empty_cache()
    else:
        device = torch.device('cpu')
    print(f"Device set to : {device}")

    print('Start testing:')
    evaluation_obj = Evaluation()
    checkpoint = torch.load(args.test_checkpoint, map_location='cpu')

    if args.eval_over_five_seeds:
        evaluation_obj.evaluate_five_seeds(checkpoint)
    else:
        evaluation_obj.evaluate(checkpoint)

if __name__ == "__main__":
    if args.is_eval:
        main_testing()
    else:
        main_training()
