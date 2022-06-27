import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
import torch.distributions as D
import math

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])

class InvDynMLP(nn.Module):
    """MLP inverse dynamics model."""

    def __init__(self, obs_dim, act_dim, mlp_w=64):
        super(InvDynMLP, self).__init__()
        # Build the model
        self.fc0 = nn.Linear(obs_dim * 2, mlp_w)
        self.fc1 = nn.Linear(mlp_w, mlp_w)
        self.fc2 = nn.Linear(mlp_w, act_dim)

    def forward(self, x):
        x = F.tanh(self.fc0(x))
        x = F.tanh(self.fc1(x))
        x = self.fc2(x)
        return x

LOG_STD_MAX = 2
LOG_STD_MIN = -20

class SquashedGaussianMLPActor(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit):
        super().__init__()
        self.net = mlp([obs_dim] + list(hidden_sizes), activation, activation)
        self.mu_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.act_limit = act_limit

    def forward(self, obs, deterministic=False, with_logprob=True):
        net_out = self.net(obs.to(device))
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        # Pre-squash distribution and sample
        pi_distribution = Normal(mu, std)
        if deterministic:
            # Only used for evaluating policy at test time.
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()

        if with_logprob:
            # Compute logprob from Gaussian, and then apply correction for Tanh squashing.
            # NOTE: The correction formula is a little bit magic. To get an understanding
            # of where it comes from, check out the original SAC paper (arXiv 1801.01290)
            # and look in appendix C. This is a more numerically-stable equivalent to Eq 21.
            # Try deriving it yourself as a (very difficult) exercise. :)
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            logp_pi -= (2*(np.log(2) - pi_action - F.softplus(-2*pi_action))).sum(axis=1)
        else:
            logp_pi = None

        pi_action = torch.tanh(pi_action)
        pi_action = self.act_limit * pi_action

        return pi_action, logp_pi

    def get_logprob(self,obs, actions):
        net_out = self.net(obs.to(device))
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)
        pi_distribution = Normal(mu, std)
        logp_pi = pi_distribution.log_prob(actions).sum(axis=-1)
        logp_pi -= (2*(np.log(2) - actions - F.softplus(-2*actions))).sum(axis=1)

        return logp_pi




class awacMLPActor(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit):
        super().__init__()
        self.net = mlp([obs_dim] + list(hidden_sizes), activation, activation)
        self.mu_layer = nn.Linear(hidden_sizes[-1], act_dim)

        self.log_std_logits = nn.Parameter(
                    torch.zeros(act_dim, requires_grad=True))
        self.min_log_std = -6
        self.max_log_std = 0
        # self.log_std_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.act_limit = act_limit

    def forward(self, obs, deterministic=False, with_logprob=True):
        # print("Using the special policy")
        net_out = self.net(obs.to(device))
        mu = self.mu_layer(net_out)
        mu = torch.tanh(mu) * self.act_limit

        log_std = torch.sigmoid(self.log_std_logits)

        log_std = self.min_log_std + log_std * (
                        self.max_log_std - self.min_log_std)
        std = torch.exp(log_std)
        # print("Std: {}".format(std))

        # Pre-squash distribution and sample
        pi_distribution = Normal(mu, std)
        if deterministic:
            # Only used for evaluating policy at test time.
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()

        if with_logprob:
            # Compute logprob from Gaussian, and then apply correction for Tanh squashing.
            # NOTE: The correction formula is a little bit magic. To get an understanding
            # of where it comes from, check out the original SAC paper (arXiv 1801.01290)
            # and look in appendix C. This is a more numerically-stable equivalent to Eq 21.
            # Try deriving it yourself as a (very difficult) exercise. :)
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            # logp_pi -= (2*(np.log(2) - pi_action - F.softplus(-2*pi_action))).sum(axis=1)
        else:
            logp_pi = None


        return pi_action, logp_pi

    def get_logprob(self,obs, actions):
        net_out = self.net(obs.to(device))
        mu = self.mu_layer(net_out)
        mu = torch.tanh(mu) * self.act_limit
        log_std = torch.sigmoid(self.log_std_logits)
        # log_std = self.log_std_layer(net_out)
        log_std = self.min_log_std + log_std * (
                        self.max_log_std - self.min_log_std)
        std = torch.exp(log_std)
        pi_distribution = Normal(mu, std)
        logp_pi = pi_distribution.log_prob(actions).sum(axis=-1)

        return logp_pi


class awacCNNActor(nn.Module):

    def __init__(self, act_dim, act_limit):
        super().__init__()

        img_size = 32
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

        self.mu_layer = nn.Linear(512, act_dim)

        self.log_std_logits = nn.Parameter(
                    torch.zeros(act_dim, requires_grad=True))
        self.min_log_std = -6
        self.max_log_std = 0
        self.act_limit = act_limit

    def forward(self, obs, deterministic=False, with_logprob=True):
        # print("Using the special policy")
        if isinstance(obs, dict):
            obs = torch.as_tensor(obs['image'], dtype=torch.float32)

        obs = self.cnn_layers(obs.to(device))
        obs = torch.flatten(obs, 1)
        net_out = self.latent_pi(obs)
        mu = self.mu_layer(net_out)
        mu = torch.tanh(mu) * self.act_limit

        log_std = torch.sigmoid(self.log_std_logits)

        log_std = self.min_log_std + log_std * (
                        self.max_log_std - self.min_log_std)
        std = torch.exp(log_std)
        # print("Std: {}".format(std))

        # Pre-squash distribution and sample
        pi_distribution = Normal(mu, std)
        if deterministic:
            # Only used for evaluating policy at test time.
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()

        if with_logprob:
            # Compute logprob from Gaussian, and then apply correction for Tanh squashing.
            # NOTE: The correction formula is a little bit magic. To get an understanding
            # of where it comes from, check out the original SAC paper (arXiv 1801.01290)
            # and look in appendix C. This is a more numerically-stable equivalent to Eq 21.
            # Try deriving it yourself as a (very difficult) exercise. :)
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            # logp_pi -= (2*(np.log(2) - pi_action - F.softplus(-2*pi_action))).sum(axis=1)
        else:
            logp_pi = None


        return pi_action, logp_pi

    def get_logprob(self,obs, actions):
        if isinstance(obs, dict):
            obs = torch.as_tensor(obs['image'], dtype=torch.float32)

        obs = self.cnn_layers(obs.to(device))
        obs = torch.flatten(obs, 1)
        net_out = self.latent_pi(obs)
        mu = self.mu_layer(net_out)
        mu = torch.tanh(mu) * self.act_limit
        log_std = torch.sigmoid(self.log_std_logits)
        log_std = self.min_log_std + log_std * (
                        self.max_log_std - self.min_log_std)
        std = torch.exp(log_std)
        pi_distribution = Normal(mu, std)
        logp_pi = pi_distribution.log_prob(actions).sum(axis=-1)

        return logp_pi

class awacDrQCNNEncoder(nn.Module):
    def __init__(self, env_image_size, img_channel, feature_dim):
        super().__init__()
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(in_channels=img_channel, out_channels=32, kernel_size=3, stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1),
            nn.LeakyReLU(),
			nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1),
            nn.LeakyReLU(),
        )

        test_mat = torch.zeros(1, img_channel, env_image_size, env_image_size)
        for conv_layer in self.cnn_layers:
            test_mat = conv_layer(test_mat)
        fc_input_size = int(np.prod(test_mat.shape))

        self.head = nn.Sequential(
            nn.Linear(fc_input_size, feature_dim),
            nn.LayerNorm(feature_dim))

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

    def forward(self, obs, detach=False):
        obs = obs / 255.
        out = self.cnn_layers(obs)

        if detach:
            out = out.detach()

        out = torch.flatten(out, 1)
        out = self.head(out)
        out = torch.tanh(out)
        return out

    def tie_weights(self, src, trg):
        assert type(src) == type(trg)
        trg.weight = src.weight
        trg.bias = src.bias


class awacDrQCNNActor(nn.Module):

    def __init__(self, act_dim, act_limit, env_image_size, img_channel, feature_dim):
        super().__init__()

        self.encoder = awacDrQCNNEncoder(env_image_size, img_channel, feature_dim)

        self.latent_pi = nn.Sequential(
            nn.Linear(feature_dim, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 1024),
            nn.LeakyReLU(),
        )

        self.mu_layer = nn.Linear(1024, act_dim)

        self.log_std_logits = nn.Parameter(
                    torch.zeros(act_dim, requires_grad=True))
        self.min_log_std = -6
        self.max_log_std = 0
        self.act_limit = act_limit

    def forward(self, obs, deterministic=False, with_logprob=True):
        if isinstance(obs, dict):
            obs = torch.as_tensor(obs['image'], dtype=torch.float32)

        obs = self.encoder(obs.to(device))
        net_out = self.latent_pi(obs)
        mu = self.mu_layer(net_out)
        mu = torch.tanh(mu) * self.act_limit

        log_std = torch.sigmoid(self.log_std_logits)

        log_std = self.min_log_std + log_std * (
                        self.max_log_std - self.min_log_std)
        std = torch.exp(log_std)
        # print("Std: {}".format(std))

        # Pre-squash distribution and sample
        pi_distribution = Normal(mu, std)
        if deterministic:
            # Only used for evaluating policy at test time.
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()

        if with_logprob:
            # Compute logprob from Gaussian, and then apply correction for Tanh squashing.
            # NOTE: The correction formula is a little bit magic. To get an understanding
            # of where it comes from, check out the original SAC paper (arXiv 1801.01290)
            # and look in appendix C. This is a more numerically-stable equivalent to Eq 21.
            # Try deriving it yourself as a (very difficult) exercise. :)
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            # logp_pi -= (2*(np.log(2) - pi_action - F.softplus(-2*pi_action))).sum(axis=1)
        else:
            logp_pi = None


        return pi_action, logp_pi

    def get_logprob(self,obs, actions):
        if isinstance(obs, dict):
            obs = torch.as_tensor(obs['image'], dtype=torch.float32)

        obs = self.encoder(obs.to(device))
        net_out = self.latent_pi(obs)
        mu = self.mu_layer(net_out)
        mu = torch.tanh(mu) * self.act_limit
        log_std = torch.sigmoid(self.log_std_logits)
        log_std = self.min_log_std + log_std * (
                        self.max_log_std - self.min_log_std)
        std = torch.exp(log_std)
        pi_distribution = Normal(mu, std)
        logp_pi = pi_distribution.log_prob(actions).sum(axis=-1)

        return logp_pi

class awacDrQCNNQFunction(nn.Module):

    def __init__(self, obs_dim, act_dim, env_image_size, img_channel, feature_dim):
        super().__init__()

        self.encoder = awacDrQCNNEncoder(env_image_size, img_channel, feature_dim)

        self.q = nn.Sequential(
            nn.Linear(feature_dim + obs_dim + act_dim, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 1),
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

    def forward(self, obs, obs_keypoint, act):
        obs = self.encoder(obs.to(device))
        obs = torch.cat([obs, obs_keypoint.to(device), act.to(device)], axis=1)
        q = self.q(obs)
        return torch.squeeze(q, -1) # Critical to ensure q has right shape.

class awacDrQCNNImageOnlyQFunction(nn.Module):

    def __init__(self, obs_dim, act_dim, env_image_size, img_channel, feature_dim):
        super().__init__()

        self.encoder = awacDrQCNNEncoder(env_image_size, img_channel, feature_dim)

        self.q = nn.Sequential(
            nn.Linear(feature_dim + act_dim, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 1),
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

    def forward(self, obs, act):
        obs = self.encoder(obs.to(device))
        obs = torch.cat([obs, act.to(device)], axis=1)
        q = self.q(obs)
        return torch.squeeze(q, -1) # Critical to ensure q has right shape.

class awacDrQCNNStateOnlyQFunction(nn.Module):

    def __init__(self, obs_dim, act_dim, env_image_size, img_channel, feature_dim):
        super().__init__()

        self.q = nn.Sequential(
            nn.Linear(obs_dim + act_dim, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 1),
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

    def forward(self, obs_keypoint, act):
        obs = torch.cat([obs_keypoint.to(device), act.to(device)], axis=1)
        q = self.q(obs)
        return torch.squeeze(q, -1) # Critical to ensure q has right shape.

class MLPVFunction(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.v = mlp([obs_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs):
        v = self.v(obs.to(device))
        return torch.squeeze(v, -1) # Critical to ensure q has right shape.

class MLPQFunction(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.q = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs, act):
        q = self.q(torch.cat([obs.to(device), act.to(device)], dim=-1))
        return torch.squeeze(q, -1) # Critical to ensure q has right shape.

class MLPActorCritic(nn.Module):

    # def __init__(self, observation_space, action_space, hidden_sizes=(256,256),
    #              activation=nn.ReLU, special_policy=None):
    def __init__(self, observation_space, action_space, hidden_sizes=(1024,1024),
                 activation=nn.ReLU, bc_model_ckpt_file=None, special_policy=None, env_image_size=32, img_channel=None):
        super().__init__()

        feature_dim = 50

        if special_policy in ['awac_img', 'awac_img_only', 'awac_state']:
            obs_dim = observation_space['key_point'].shape[0]
        else:
            obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        try:
            act_limit = action_space.high[0]
        except:
            # there is no high attribute in action_space
            act_limit = 1
        # build policy and value functions
        if special_policy is 'awac':
            # self.pi = awacMLPActor(obs_dim, act_dim, (256,256,256,256), activation, act_limit).to(device)
            self.pi = awacMLPActor(obs_dim, act_dim, hidden_sizes, activation, act_limit).to(device)
        elif special_policy in ['awac_img', 'awac_img_only', 'awac_state']:
            self.pi = awacDrQCNNActor(act_dim, act_limit, env_image_size, img_channel, feature_dim).to(device)
        else:
            self.pi = SquashedGaussianMLPActor(obs_dim, act_dim, hidden_sizes, activation, act_limit).to(device)

        if bc_model_ckpt_file:
            awac_to_bc_keys_mapping = {
                'net.0.weight': 'latent_pi.0.weight',
                'net.0.bias': 'latent_pi.0.bias',
                'net.2.weight': 'latent_pi.2.weight',
                'net.2.bias': 'latent_pi.2.bias',
                'mu_layer.weight': 'mu.0.weight',
                'mu_layer.bias': 'mu.0.bias',
                'log_std_logits': None,
                'cnn_layers.0.weight': 'cnn_layers.0.weight',
                'cnn_layers.0.bias': 'cnn_layers.0.bias',
                'cnn_layers.2.weight': 'cnn_layers.2.weight',
                'cnn_layers.2.bias': 'cnn_layers.2.bias',
                'cnn_layers.4.weight': 'cnn_layers.4.weight',
                'cnn_layers.4.bias': 'cnn_layers.4.bias',
                'latent_pi.0.weight': 'latent_pi.0.weight',
                'latent_pi.0.bias': 'latent_pi.0.bias',
                'latent_pi.2.weight': 'latent_pi.2.weight',
                'latent_pi.2.bias': 'latent_pi.2.bias',
            }
            bc_pretrained_state_dict = torch.load(bc_model_ckpt_file)['state_dict']
            pretrained_dict = dict()
            for k in self.pi.state_dict().keys():
                bc_key = awac_to_bc_keys_mapping[k]
                if bc_key:
                    pretrained_dict[k] = bc_pretrained_state_dict[bc_key]
                else:
                    pretrained_dict[k] = self.pi.state_dict()[k]
            self.pi.load_state_dict(pretrained_dict)

        if special_policy is 'awac_img':
            self.q1 = awacDrQCNNQFunction(obs_dim, act_dim, env_image_size, img_channel, feature_dim).to(device)
            self.q2 = awacDrQCNNQFunction(obs_dim, act_dim, env_image_size, img_channel, feature_dim).to(device)
        elif special_policy is 'awac_img_only':
            self.q1 = awacDrQCNNImageOnlyQFunction(obs_dim, act_dim, env_image_size, img_channel, feature_dim).to(device)
            self.q2 = awacDrQCNNImageOnlyQFunction(obs_dim, act_dim, env_image_size, img_channel, feature_dim).to(device)
        elif special_policy is 'awac_state':
            self.q1 = awacDrQCNNStateOnlyQFunction(obs_dim, act_dim, env_image_size, img_channel, feature_dim).to(device)
            self.q2 = awacDrQCNNStateOnlyQFunction(obs_dim, act_dim, env_image_size, img_channel, feature_dim).to(device)
        else:
            self.q1 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation).to(device)
            self.q2 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation).to(device)


    def act_batch(self, obs, deterministic=False):
        with torch.no_grad():
            a, _ = self.pi(obs, deterministic, False)
            return a

    def act(self, obs, deterministic=False):
        with torch.no_grad():
            a, _ = self.pi(obs, deterministic, False)
            return a.cpu().data.numpy().flatten()





# --------------------
# Density estimator
# Model layers and helpers
# --------------------

def create_masks(input_size, hidden_size, n_hidden, input_order='sequential', input_degrees=None):
    # MADE paper sec 4:
    # degrees of connections between layers -- ensure at most in_degree - 1 connections
    degrees = []

    # set input degrees to what is provided in args (the flipped order of the previous layer in a stack of mades);
    # else init input degrees based on strategy in input_order (sequential or random)
    if input_order == 'sequential':
        degrees += [torch.arange(input_size)] if input_degrees is None else [input_degrees]
        for _ in range(n_hidden + 1):
            degrees += [torch.arange(hidden_size) % (input_size - 1)]
        degrees += [torch.arange(input_size) % input_size - 1] if input_degrees is None else [input_degrees % input_size - 1]

    elif input_order == 'random':
        degrees += [torch.randperm(input_size)] if input_degrees is None else [input_degrees]
        for _ in range(n_hidden + 1):
            min_prev_degree = min(degrees[-1].min().item(), input_size - 1)
            degrees += [torch.randint(min_prev_degree, input_size, (hidden_size,))]
        min_prev_degree = min(degrees[-1].min().item(), input_size - 1)
        degrees += [torch.randint(min_prev_degree, input_size, (input_size,)) - 1] if input_degrees is None else [input_degrees - 1]

    # construct masks
    masks = []
    for (d0, d1) in zip(degrees[:-1], degrees[1:]):
        masks += [(d1.unsqueeze(-1) >= d0.unsqueeze(0)).float()]

    return masks, degrees[0]


class MaskedLinear(nn.Linear):
    """ MADE building block layer """
    def __init__(self, input_size, n_outputs, mask, cond_label_size=None):
        super().__init__(input_size, n_outputs)

        self.register_buffer('mask', mask)

        self.cond_label_size = cond_label_size
        if cond_label_size is not None:
            self.cond_weight = nn.Parameter(torch.rand(n_outputs, cond_label_size) / math.sqrt(cond_label_size))

    def forward(self, x, y=None):
        out = F.linear(x, self.weight * self.mask, self.bias)
        if y is not None:
            out = out + F.linear(y, self.cond_weight)
        return out

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        ) + (self.cond_label_size != None) * ', cond_features={}'.format(self.cond_label_size)


class MADE(nn.Module):
    def __init__(self, input_size, hidden_size, n_hidden, cond_label_size=None, activation='relu', input_order='sequential', input_degrees=None):
        """
        Args:
            input_size -- scalar; dim of inputs
            hidden_size -- scalar; dim of hidden layers
            n_hidden -- scalar; number of hidden layers
            activation -- str; activation function to use
            input_order -- str or tensor; variable order for creating the autoregressive masks (sequential|random)
                            or the order flipped from the previous layer in a stack of mades
            conditional -- bool; whether model is conditional
        """
        super().__init__()
        # base distribution for calculation of log prob under the model
        self.register_buffer('base_dist_mean', torch.zeros(input_size))
        self.register_buffer('base_dist_var', torch.ones(input_size))

        # create masks
        masks, self.input_degrees = create_masks(input_size, hidden_size, n_hidden, input_order, input_degrees)

        # setup activation
        if activation == 'relu':
            activation_fn = nn.ReLU()
        elif activation == 'tanh':
            activation_fn = nn.Tanh()
        else:
            raise ValueError('Check activation function.')

        # construct model
        self.net_input = MaskedLinear(input_size, hidden_size, masks[0], cond_label_size)
        self.net = []
        for m in masks[1:-1]:
            self.net += [activation_fn, MaskedLinear(hidden_size, hidden_size, m)]
        self.net += [activation_fn, MaskedLinear(hidden_size, 2 * input_size, masks[-1].repeat(2,1))]
        self.net = nn.Sequential(*self.net)

    @property
    def base_dist(self):
        return D.Normal(self.base_dist_mean, self.base_dist_var)

    def forward(self, x, y=None):
        # MAF eq 4 -- return mean and log std
        m, loga = self.net(self.net_input(x, y)).chunk(chunks=2, dim=1)
        u = (x - m) * torch.exp(-loga)
        # MAF eq 5
        log_abs_det_jacobian = - loga
        return u, log_abs_det_jacobian

    def inverse(self, u, y=None, sum_log_abs_det_jacobians=None):
        # MAF eq 3
        D = u.shape[1]
        x = torch.zeros_like(u)
        # run through reverse model
        for i in self.input_degrees:
            m, loga = self.net(self.net_input(x, y)).chunk(chunks=2, dim=1)
            x[:,i] = u[:,i] * torch.exp(loga[:,i]) + m[:,i]
        log_abs_det_jacobian = loga
        return x, log_abs_det_jacobian

    def log_prob(self, x, y=None):
        u, log_abs_det_jacobian = self.forward(x, y)
        return torch.sum(self.base_dist.log_prob(u) + log_abs_det_jacobian, dim=1)
