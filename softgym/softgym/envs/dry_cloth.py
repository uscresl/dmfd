import numpy as np
import pyflex
from copy import deepcopy
from softgym.envs.cloth_env import ClothEnv
from softgym.utils.misc import quatFromAxisAngle
from softgym.utils.pyflex_utils import center_object


class DryClothEnv(ClothEnv):
    def __init__(self, cached_states_path='dry_cloth_init_states.pkl', **kwargs):
        self.fold_group_a = self.fold_group_b = None
        self.init_pos, self.prev_dist = None, None

        self.default_box_height = 0.3
        self.rack_center = -0.4

        super().__init__(**kwargs)
        assert self.action_mode == 'pickerpickandplace' # we only support pickerpickandplace action mode
        self.get_cached_configs_and_states(cached_states_path, self.num_variations)

        # RSI and IR
        self.enable_rsi = kwargs.get('enable_rsi', False)
        self.enable_rsi_ir = False
        self.rsi_ir_prob = kwargs.get('rsi_ir_prob', 0)
        rsi_file = kwargs.get('rsi_file', None)
        if rsi_file is not None:
            self.reference_next_state_info = np.load(rsi_file, allow_pickle=True)
        self.reference_next_state_info_ep = None
        self.reference_next_action_info_ep = None
        self.chosen_step = 0
        self.non_rsi_ir = kwargs.get('non_rsi_ir', False)
        self.enable_action_matching = kwargs.get('enable_action_matching', False)
        self.enable_loading_states_from_folder = kwargs.get('enable_loading_states_from_folder', False)
        self.ep_task_reward = self.ep_il_reward = 0. # initialize

        # expert policy
        self.expert_policy_index = 0
        # implementation with picker action [backup]
        # self.human_expert_actions = np.array([
        #     [1, -1, 0.5, 1, 1, -1, -0.5, 1, \
        #     -1, 0.85, 0.5, 0, -1, 0.85, -0.5, 0], # go to corner and pick up cloth & move to top of rack and drop cloth
        #     [0, 0, 0.2, 0, 0, 0, -0.2, 0, \
        #     0, 0, 0.2, 0, 0, 0, -0.2, 0] # go to starting position
        # ]) # normalized_performance: 0.971342533826828
        self.human_expert_actions = np.array([
            [-0.44, -1, 0.5, -0.44, -1, -0.5, \
            -1, 0.7, 0.5, -1, 0.7, -0.5], # go to corner and pick up cloth & move to top of rack and drop cloth
            [0, 0, 0.2, 0, 0, -0.2, \
            0, 0, 0.2, 0, 0, -0.2] # go to starting position
        ]) # normalized_performance: 0.94

    def get_default_config(self):
        """ Set the default config of the environment and load it to self.config """
        particle_radius = self.cloth_particle_radius
        if self.action_mode in ['sawyer', 'franka']:
            cam_pos, cam_angle = np.array([0.0, 1.62576, 1.04091]), np.array([0.0, -0.844739, 0])
        else:
            # cam_pos, cam_angle = np.array([-0.0, 2, 2]), np.array([0, -45 / 180. * np.pi, 0.]) # side view
            cam_pos, cam_angle = np.array([-0.0, 2, 0]), np.array([0, -90 / 180. * np.pi, 0.]) # top down view
        config = {
            'ClothPos': [-1.6, 2.0, -0.8],
            'ClothSize': [int(0.6 / particle_radius), int(0.368 / particle_radius)],
            'ClothStiff': [0.8, 1, 0.9],  # Stretch, Bend and Shear
            'camera_name': 'default_camera',
            'camera_params': {'default_camera':
                                  {'pos': cam_pos,
                                   'angle': cam_angle,
                                   'width': self.camera_width,
                                   'height': self.camera_height}},
            'flip_mesh': 0,
        }

        return config

    def rotate_particles(self, angle):
        pos = pyflex.get_positions().reshape(-1, 4)
        center = np.mean(pos, axis=0)
        pos -= center
        new_pos = pos.copy()
        new_pos[:, 0] = (np.cos(angle) * pos[:, 0] - np.sin(angle) * pos[:, 2])
        new_pos[:, 2] = (np.sin(angle) * pos[:, 0] + np.cos(angle) * pos[:, 2])
        new_pos += center
        pyflex.set_positions(new_pos)

    def _sample_cloth_size(self):
        return 80, 80

    def generate_env_variation(self, num_variations=2, vary_cloth_size=True):
        """ Generate initial states. Note: This will also change the current states! """
        max_wait_step = 1000  # Maximum number of steps waiting for the cloth to stablize
        stable_vel_threshold = 0.2  # Cloth stable when all particles' vel are smaller than this
        generated_configs, generated_states = [], []
        default_config = self.get_default_config()
        default_config['flip_mesh'] = 1

        for i in range(num_variations):
            config = deepcopy(default_config)
            self.update_camera(config['camera_name'], config['camera_params'][config['camera_name']])
            if vary_cloth_size:
                cloth_dimx, cloth_dimy = self._sample_cloth_size()
                config['ClothSize'] = [cloth_dimx, cloth_dimy]
            else:
                cloth_dimx, cloth_dimy = config['ClothSize']

            self.set_scene(config)
            self.action_tool.reset([0., -1., 0.])
            pos = pyflex.get_positions().reshape(-1, 4)
            pos[:, :3] -= np.mean(pos, axis=0)[:3]
            if self.action_mode in ['sawyer', 'franka']: # Take care of the table in robot case
                pos[:, 1] = 0.57
            else:
                pos[:, 1] = 0.005
            pos[:, 3] = 1
            pyflex.set_positions(pos.flatten())
            pyflex.set_velocities(np.zeros_like(pos))
            for _ in range(5):  # In case if the cloth starts in the air
                pyflex.step()

            for wait_i in range(max_wait_step):
                pyflex.step()
                curr_vel = pyflex.get_velocities()
                if np.alltrue(np.abs(curr_vel) < stable_vel_threshold):
                    break

            center_object()
            # angle = (np.random.random() - 0.5) * np.pi / 2
            # self.rotate_particles(angle)

            generated_configs.append(deepcopy(config))
            print('config {}: {}'.format(i, config['camera_params']))
            generated_states.append(deepcopy(self.get_state()))

        return generated_configs, generated_states

    def set_test_color(self, num_particles):
        """
        Assign random colors to group a and the same colors for each corresponding particle in group b
        :return:
        """
        colors = np.zeros((num_particles))
        rand_size = 30
        rand_colors = np.random.randint(0, 5, size=rand_size)
        rand_index = np.random.choice(range(len(self.fold_group_a)), rand_size)
        colors[self.fold_group_a[rand_index]] = rand_colors
        colors[self.fold_group_b[rand_index]] = rand_colors
        self.set_colors(colors)

    def reset(self, config=None, initial_state=None, config_id=None, is_eval=False):
        if config is None:
            if config_id is None:
                if self.eval_flag:
                    eval_beg = int(0.8 * len(self.cached_configs))
                    config_id = np.random.randint(low=eval_beg, high=len(self.cached_configs)) if not self.deterministic else eval_beg
                else:
                    train_high = int(0.8 * len(self.cached_configs))
                    config_id = np.random.randint(low=0, high=max(train_high, 1)) if not self.deterministic else 0

            self.current_config = self.cached_configs[config_id]
            self.current_config_id = config_id
            self.set_scene(self.cached_configs[config_id], self.cached_init_states[config_id])
        else:
            self.current_config = config
            self.set_scene(config, initial_state)
        self.particle_num = pyflex.get_n_particles()
        self.prev_reward = 0.
        self.ep_il_reward = 0.
        self.ep_task_reward = 0.
        self.time_step = 0
        self.chosen_step = 0
        self.expert_policy_index = 0

        if (not is_eval) and self.enable_rsi and np.random.uniform(0,1) <= self.rsi_ir_prob:
            self.reset_to_state() # reset chosen_step and chosen state
            self.enable_rsi_ir = True
        else:
            self.enable_rsi_ir = False

        obs = self._reset()
        if self.recording:
            self.video_frames.append(self.render(mode='rgb_array'))
        return obs

    def reset_to_state(self):
        '''
        RSI
        '''
        state_trajs = self.reference_next_state_info['state_trajs']
        configs = self.reference_next_state_info['configs']
        ob_trajs = self.reference_next_state_info['ob_trajs']
        action_trajs = self.reference_next_state_info['action_trajs']
        reference_state_index = np.random.randint(0, len(ob_trajs))
        self.reference_next_state_info_ep = ob_trajs[reference_state_index]
        self.reference_next_action_info_ep = action_trajs[reference_state_index]
        self.chosen_step = np.random.randint(0, len(self.reference_next_state_info_ep))
        self.time_step = self.chosen_step

        # reset environment
        self.current_config = configs[reference_state_index]
        if self.enable_loading_states_from_folder:
            self.set_scene(configs[reference_state_index], np.load(state_trajs[reference_state_index][self.chosen_step], allow_pickle=True).item())
        else:
            self.set_scene(configs[reference_state_index], state_trajs[reference_state_index][self.chosen_step])

    def _reset(self):
        """ Right now only use one initial state. Need to make sure _reset always give the same result. Otherwise CEM will fail."""
        if hasattr(self, 'action_tool'):
            particle_pos = pyflex.get_positions().reshape(-1, 4)
            p1, p2, p3, p4 = self._get_key_point_idx()
            key_point_pos = particle_pos[(p1, p2), :3] # Was changed from from p1, p4.
            middle_point = np.mean(key_point_pos, axis=0)
            self.action_tool.reset([middle_point[0], 0.1, middle_point[2]])

        config = self.get_current_config()
        num_particles = np.prod(config['ClothSize'], dtype=int)
        particle_grid_idx = np.array(list(range(num_particles))).reshape(config['ClothSize'][1], config['ClothSize'][0])  # Reversed index here

        cloth_dimx = config['ClothSize'][0]
        pos = pyflex.get_positions().reshape(-1,4)
        self.xlen, self.zlen = np.abs(pos[0] - pos[-1])[0::2]

        x_split_l = int(cloth_dimx * 0.4) # left x%
        x_split_r = int(cloth_dimx * (1 - 0.4)) # right x%

        self.fold_group_a = particle_grid_idx[:, :x_split_l].flatten()
        self.fold_group_b = particle_grid_idx[:, x_split_r:].flatten()

        self.group_a_corners = np.array([particle_grid_idx[0, 0], particle_grid_idx[-1, 0], particle_grid_idx[0, x_split_l], particle_grid_idx[-1, x_split_l]]) # idx 0,1 same column. idx 2,3 same column
        self.group_b_corners = np.array([particle_grid_idx[0, x_split_r], particle_grid_idx[-1, x_split_r], particle_grid_idx[0, -1], particle_grid_idx[-1, -1]]) # idx 0,1 same column. idx 2,3 same column

        colors = np.zeros(num_particles)
        colors[self.fold_group_a] = 1
        # self.set_colors(colors) # TODO the phase actually changes the cloth dynamics so we do not change them for now. Maybe delete this later.

        pyflex.step()
        self.init_pos = pyflex.get_positions().reshape((-1, 4))[:, :3]

        self.performance_init = None
        info = self._get_info()
        self.performance_init = info['performance']

        # drying rack
        halfEdge = np.array([0.01, self.default_box_height, 0.6])
        center = np.array([self.rack_center, 0, 0])
        quat = quatFromAxisAngle([0, 0, -1.], 0.)
        pyflex.add_box(halfEdge, center, quat)
        pyflex.step()
        pyflex.set_box_color([0.7, 0.5, 0.2])

        return self._get_obs()

    def step(self, action, record_continuous_video=False, img_size=None):
        """ If record_continuous_video is set to True, will record an image for each sub-step"""
        frames = []
        for i in range(self.action_repeat):
            self._step(action)
            if record_continuous_video and i % 2 == 0:  # No need to record each step
                frames.append(self.get_image(img_size, img_size))
        obs = self._get_obs()
        reward, rew_info = self.compute_reward(action, obs, set_prev_reward=True, rew_info=True)
        info = self._get_info()
        self.ep_il_reward += rew_info['il_reward']
        self.ep_task_reward += rew_info['task_reward']

        if self.recording:
            self.video_frames.append(self.render(mode='rgb_array'))
        self.chosen_step += 1
        self.time_step += 1

        done = False
        if self.time_step >= self.horizon:
            done = True
        if record_continuous_video:
            info['flex_env_recorded_frames'] = frames
        return obs, reward, done, info

    def _step(self, action):
        self.action_tool.step(action)
        if self.action_mode in ['sawyer', 'franka']:
            print(self.action_tool.next_action)
            pyflex.step(self.action_tool.next_action)
        else:
            pyflex.step()

    def _get_obs_key_points(self):
        particle_pos = np.array(pyflex.get_positions()).reshape([-1, 4])[:, :3]
        keypoint_pos = particle_pos[self._get_key_point_idx(), :3]
        pos = keypoint_pos

        if self.action_mode == 'pickerpickandplace':
            shapes = pyflex.get_shape_states()
            shapes = np.reshape(shapes, [-1, 14])
            # extract actuators' shape and exclude rack in DryCloth
            pos = np.concatenate([pos.flatten(), shapes[:self.action_tool.num_picker, 0:3].flatten()])
        return pos

    def compute_task_reward(self, rew_info=False):
        """
        The particles are split into three groups, (a, b, and other).
        Group a should be right of the drying rack, and above the ground.
        Group b should be left of the drying rack, and above the ground.
        """
        #TODO: Will zdist really help? It will make it worse for the one-handed picker

        pos = pyflex.get_positions()
        pos = pos.reshape((-1, 4))[:, :3]
        pos_group_a = pos[self.fold_group_a]
        pos_group_b = pos[self.fold_group_b]
        x0 = self.rack_center
        y0 = 0.05

        # Get number of particles that are greater than x0 and above y0
        xgoal = pos_group_a[:, 0] < x0
        ygoal = pos_group_a[:, 1] > y0
        num_a_particles_in_goal = np.sum(xgoal & ygoal) # note:bitwise and
        frac_a_particles_in_goal = num_a_particles_in_goal / len(pos_group_a)

        zdist1 = pos[self.group_a_corners[0]][2] - pos[self.group_a_corners[1]][2]
        zdist2 = pos[self.group_a_corners[2]][2] - pos[self.group_a_corners[3]][2]
        zdist = np.mean([np.abs(zdist1), np.abs(zdist2)])
        zgoal_a_quality = -np.abs(zdist - self.zlen)

        # Get number of particles that are less than x0 and above y0
        xgoal = pos_group_b[:, 0] > x0
        ygoal = pos_group_b[:, 1] > y0
        num_b_particles_in_goal = np.sum(xgoal & ygoal) # note:bitwise and
        frac_b_particles_in_goal = num_b_particles_in_goal / len(pos_group_b)

        zdist1 = pos[self.group_b_corners[0]][2] - pos[self.group_b_corners[1]][2]
        zdist2 = pos[self.group_b_corners[2]][2] - pos[self.group_b_corners[3]][2]
        zdist = np.mean([np.abs(zdist1), np.abs(zdist2)])
        zgoal_b_quality = -np.abs(zdist - self.zlen)

        performance = np.mean([frac_a_particles_in_goal, frac_b_particles_in_goal]) + np.mean([zgoal_a_quality, zgoal_b_quality])
        if rew_info:
            rew_dict = {
                'task_reward': performance,
                'frac_a_particles_in_goal': frac_a_particles_in_goal,
                'frac_b_particles_in_goal': frac_b_particles_in_goal,
                'zgoal_a_quality': zgoal_a_quality,
                'zgoal_b_quality': zgoal_b_quality,
                }
            return performance, rew_dict
        else:
            return performance

    def compute_reward(self, action=None, obs=None, set_prev_reward=False, expert_action=None, rew_info=False):
        """
        :param pos: nx4 matrix (x, y, z, inv_mass), the standard for extended position-based dynamics (XPBD)
        """
        performance = self.compute_task_reward()
        reward = performance

        # Imitation Rewards
        if self.enable_rsi_ir and \
            self.chosen_step < len(self.reference_next_state_info_ep):
            obs_key_points = self._get_obs_key_points()
            ref_dist = np.linalg.norm(self.reference_next_state_info_ep[self.chosen_step] - obs_key_points)

            # il_reward = np.exp(-1 * ref_dist) # range [0,1], disable due to nan issue when multiplication is used in np.exp
            # sigmoid with cutoff at x=0 because ref_dist cannot be negative, so range is [0, 0.5]
            il_reward = 1/(1 + np.exp(ref_dist))
        else:
            il_reward = 0
        reward += il_reward
        if rew_info:
            return reward, {'il_reward': il_reward, 'task_reward': performance}
        else:
            return reward

    def _get_info(self):
        performance, rew_dict = self.compute_task_reward(rew_info=True)
        performance_init = performance if self.performance_init is None else self.performance_init  # Use the original performance
        info = {
            'performance': performance,
            'normalized_performance': (performance - performance_init) / (1. - performance_init),
            'frac_a_particles_in_goal': rew_dict['frac_a_particles_in_goal'],
            'frac_b_particles_in_goal': rew_dict['frac_b_particles_in_goal'],
            'zgoal_a_quality': rew_dict['zgoal_a_quality'],
            'zgoal_b_quality': rew_dict['zgoal_b_quality'],
        }
        if 'qpg' in self.action_mode:
            info['total_steps'] = self.action_tool.total_steps
        return info

    def _set_to_folded(self):
        config = self.get_current_config()
        num_particles = np.prod(config['ClothSize'], dtype=int)
        particle_grid_idx = np.array(list(range(num_particles))).reshape(config['ClothSize'][1], config['ClothSize'][0])  # Reversed index here

        cloth_dimx = config['ClothSize'][0]
        x_split = cloth_dimx // 2
        fold_group_a = particle_grid_idx[:, :x_split].flatten()
        fold_group_b = np.flip(particle_grid_idx, axis=1)[:, :x_split].flatten()

        curr_pos = pyflex.get_positions().reshape((-1, 4))
        curr_pos[fold_group_a, :] = curr_pos[fold_group_b, :].copy()
        curr_pos[fold_group_a, 1] += 0.05  # group a particle position made tcurr_pos[self.fold_group_b, 1] + 0.05e at top of group b position.

        pyflex.set_positions(curr_pos)
        for i in range(10):
            pyflex.step()
        return self._get_info()['performance']

    def compute_expert_action(self):
        if self.expert_policy_index < self.human_expert_actions.shape[0]:
            action = self.human_expert_actions[self.expert_policy_index]
            self.expert_policy_index += 1
        else:
            action = self.human_expert_actions[-1]

        return action
