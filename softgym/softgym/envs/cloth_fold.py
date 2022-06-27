import numpy as np
import pyflex
from copy import deepcopy
from softgym.envs.cloth_env import ClothEnv
from softgym.utils.pyflex_utils import center_object


class ClothFoldEnv(ClothEnv):
    def __init__(self, cached_states_path='cloth_fold_init_states.pkl', **kwargs):
        self.fold_group_a = self.fold_group_b = None
        self.init_pos, self.prev_dist = None, None
        self.num_variations = kwargs['num_variations']
        super().__init__(**kwargs)
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
        if self.num_variations == 1:
            return 80, 80
        else:
            return np.random.randint(60, 120), np.random.randint(60, 120)

    def get_default_config(self):
        """ Set the default config of the environment and load it to self.config """
        particle_radius = self.cloth_particle_radius
        if self.action_mode in ['sawyer', 'franka']:
            cam_pos, cam_angle = np.array([0.0, 1.62576, 1.04091]), np.array([0.0, -0.844739, 0])
        else:
            if self.num_variations == 1:
                # top down view only for sim-to-real experiments
                cam_pos, cam_angle = np.array([-0.0, 1.0, 0]), np.array([0, -90 / 180. * np.pi, 0.])
            else:
                 # default side view
                cam_pos, cam_angle = np.array([-0.0, 0.82, 0.82]), np.array([0, -45 / 180. * np.pi, 0.])
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
            angle = (np.random.random() - 0.5) * np.pi / 2
            if self.num_variations != 1:
                self.rotate_particles(angle)

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

            # picker_low = self.action_tool.picker_low
            # picker_high = self.action_tool.picker_high
            # offset_x = self.action_tool._get_pos()[0][0][0] - picker_low[0] - 0.3
            # picker_low[0] += offset_x
            # picker_high[0] += offset_x
            # picker_high[0] += 1.0
            # self.action_tool.update_picker_boundary(picker_low, picker_high)

        config = self.get_current_config()
        num_particles = np.prod(config['ClothSize'], dtype=int)
        particle_grid_idx = np.array(list(range(num_particles))).reshape(config['ClothSize'][1], config['ClothSize'][0])  # Reversed index here

        cloth_dimx = config['ClothSize'][0]
        x_split = cloth_dimx // 2
        self.fold_group_a = particle_grid_idx[:, :x_split].flatten()
        self.fold_group_b = np.flip(particle_grid_idx, axis=1)[:, :x_split].flatten()

        colors = np.zeros(num_particles)
        colors[self.fold_group_a] = 1
        # self.set_colors(colors) # TODO the phase actually changes the cloth dynamics so we do not change them for now. Maybe delete this later.

        pyflex.step()
        self.init_pos = pyflex.get_positions().reshape((-1, 4))[:, :3]
        pos_a = self.init_pos[self.fold_group_a, :]
        pos_b = self.init_pos[self.fold_group_b, :]
        self.prev_dist = np.mean(np.linalg.norm(pos_a - pos_b, axis=1))

        self.performance_init = None
        info = self._get_info()
        self.performance_init = info['performance']

        self.x_min = np.amin(particle_pos[:, 0])
        self.x_max = np.amax(particle_pos[:, 0])

        self.z_min = np.amin(particle_pos[:, 2])
        self.z_max = np.amax(particle_pos[:, 2])

        self.y_coor = particle_pos[0, 1] # checked and saw that these were all the same


        corner1_idx = particle_grid_idx[0, 0]
        corner2_idx = particle_grid_idx[0, -1]
        corner3_idx = particle_grid_idx[-1, 0]
        corner4_idx = particle_grid_idx[-1, -1]

        particle_pos[particle_grid_idx[0, 0]]
        particle_pos[particle_grid_idx[0, -1]]
        particle_pos[particle_grid_idx[-1, 0]]
        particle_pos[particle_grid_idx[-1, -1]]

        self.corner_pick_idx = []
        self.corner_drop_idx = []

        # test to see which cloth folding group to place each corner in
        if corner1_idx in self.fold_group_a:
            self.corner_pick_idx.append(corner1_idx)
        else:
            self.corner_drop_idx.append(corner1_idx)

        if corner2_idx in self.fold_group_a:
            self.corner_pick_idx.append(corner2_idx)
        else:
            self.corner_drop_idx.append(corner2_idx)

        if corner3_idx in self.fold_group_a:
            self.corner_pick_idx.append(corner3_idx)
        else:
            self.corner_drop_idx.append(corner3_idx)

        if corner4_idx in self.fold_group_a:
            self.corner_pick_idx.append(corner4_idx)
        else:
            self.corner_drop_idx.append(corner4_idx)

        corner_pick0 = particle_pos[self.corner_pick_idx[0]][:3]
        corner_drop0 = particle_pos[self.corner_drop_idx[0]][:3]
        corner_drop1 = particle_pos[self.corner_drop_idx[1]][:3]

        # calculate distance between corner_pick and corner_drop, corner_drop
        dist0 = np.linalg.norm(corner_pick0 - corner_drop0)
        dist1 = np.linalg.norm(corner_pick0 - corner_drop1)

        # if this is the case swap so we don't fold diagonally
        if dist0 > dist1:
            tmp = self.corner_drop_idx[0]
            self.corner_drop_idx[0] = self.corner_drop_idx[1]
            self.corner_drop_idx[1] = tmp

        self.expert_state = 0

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

        if self.action_mode in ['sphere', 'picker', 'pickerpickandplace']:
            shapes = pyflex.get_shape_states()
            shapes = np.reshape(shapes, [-1, 14])
            pos = np.concatenate([pos.flatten(), shapes[:, 0:3].flatten()])
        return pos

    def compute_reward(self, action=None, obs=None, set_prev_reward=False, expert_action=None, rew_info=False):
        """
        The particles are splitted into two groups. The reward will be the minus average eculidean distance between each
        particle in group a and the crresponding particle in group b
        :param pos: nx4 matrix (x, y, z, inv_mass)
        """
        pos = pyflex.get_positions()
        pos = pos.reshape((-1, 4))[:, :3]
        pos_group_a = pos[self.fold_group_a]
        pos_group_b = pos[self.fold_group_b]
        pos_group_b_init = self.init_pos[self.fold_group_b]
        curr_dist = np.mean(np.linalg.norm(pos_group_a - pos_group_b, axis=1)) + \
                    1.2 * np.mean(np.linalg.norm(pos_group_b - pos_group_b_init, axis=1))
        reward = -curr_dist

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
            return reward, {'il_reward': il_reward, 'task_reward': -curr_dist}

        return reward

    def _get_info(self):
        # Duplicate of the compute reward function!
        pos = pyflex.get_positions()
        pos = pos.reshape((-1, 4))[:, :3]
        pos_group_a = pos[self.fold_group_a]
        pos_group_b = pos[self.fold_group_b]
        pos_group_b_init = self.init_pos[self.fold_group_b]
        group_dist = np.mean(np.linalg.norm(pos_group_a - pos_group_b, axis=1))
        fixation_dist = np.mean(np.linalg.norm(pos_group_b - pos_group_b_init, axis=1))
        performance = -group_dist - 1.2 * fixation_dist
        performance_init = performance if self.performance_init is None else self.performance_init  # Use the original performance
        info = {
            'performance': performance,
            'normalized_performance': (performance - performance_init) / (0. - performance_init),
            'neg_group_dist': -group_dist,
            'neg_fixation_dist': -fixation_dist
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
        """ Simple (suboptimal) expert: Pick two corners and move towards the other two.
        """
        if self.action_mode == 'pickerpickandplace':
            if self.action_tool.num_picker == 2:
                particle_pos = pyflex.get_positions().reshape(-1, 4)
                pick1 = particle_pos[self.corner_pick_idx[0]][:3]
                pick2 = particle_pos[self.corner_pick_idx[1]][:3]
                pick1[1] = 0.01
                pick2[1] = 0.01

                place1 = particle_pos[self.corner_drop_idx[0]][:3]
                place2 = particle_pos[self.corner_drop_idx[1]][:3]
                place1[1] = 0.1
                place2[1] = 0.1
                action = np.concatenate((pick1, pick2, place1, place2))
                expert_action = self.normalize_action(action)
            elif self.action_tool.num_picker == 1:
                particle_pos = pyflex.get_positions().reshape(-1, 4)
                if self.expert_state % 2 == 0:
                    pick1 = particle_pos[self.corner_pick_idx[0]][:3]
                    pick1[1] = 0.01
                    place1 = particle_pos[self.corner_drop_idx[0]][:3]
                    place1[1] = 0.12
                    action = np.concatenate((pick1, place1))
                else:
                    pick2 = particle_pos[self.corner_pick_idx[1]][:3]
                    pick2[1] = 0.01
                    place2 = particle_pos[self.corner_drop_idx[1]][:3]
                    place2[1] = 0.12
                    action = np.concatenate((pick2, place2))
                expert_action = self.normalize_action(action)
                self.expert_state += 1
            else:
                raise NotImplementedError
        else:
            picker_pos, particle_pos = self.action_tool._get_pos()
            # locate corners of particle_pos and then do 4 picks instead of two
            pick1 = picker_pos[0, :3]
            pick2 = picker_pos[-1, :3]

            do_pick_thresh = self.action_tool.picker_radius + self.action_tool.particle_radius + self.action_tool.picker_threshold

            # go to the corner
            if self.expert_state == 0:
                end1 = self.corner_pick[0]
                end2 = self.corner_pick[1]
                pick1_xz = np.array([pick1[0]] + [pick1[2]])
                pick2_xz = np.array([pick2[0]] + [pick2[2]])

                end1_xz = np.array([end1[0]] + [end1[2]])
                end2_xz = np.array([end2[0]] + [end2[2]])

                if np.linalg.norm(abs(pick1_xz - end1_xz)) < 0.01 or \
                    np.linalg.norm(abs(pick2_xz - end2_xz)) < 0.01:
                    # print(f'Switch to 1\nend1: {end1}\nend2: {end2}\npick1: {pick1}\npick2: {pick2}')
                    self.expert_state = 1

            # go down to the corner
            elif self.expert_state == 1:
                end1 = self.corner_pick[0]
                end2 = self.corner_pick[1]

                if np.linalg.norm(abs(pick1 - end1)) < do_pick_thresh or \
                    np.linalg.norm(abs(pick2 - end2)) < do_pick_thresh:
                    # print(f'Switch to 2\nend1: {end1}\nend2: {end2}\npick1: {pick1}\npick2: {pick2}')
                    self.expert_state = 2

            # pick up
            elif self.expert_state == 2:
                end1 = self.corner_pick[0]
                end2 = self.corner_pick[1]

                end2[1] = 0.2
                end1[1] = 0.2

                if np.linalg.norm(abs(pick1 - end1)) < do_pick_thresh or \
                    np.linalg.norm(abs(pick2 - end2)) < do_pick_thresh:
                    # print(f'Switch to 3\nend1: {end1}\nend2: {end2}\npick1: {pick1}\npick2: {pick2}')
                    self.expert_state = 3

            # go to the goal corner
            elif self.expert_state == 3:
                # end1 = self.corner_drop[0]
                # end2 = self.corner_drop[1]
                end1 = particle_pos[self.corner_drop_idx[0]][:3]
                end2 = particle_pos[self.corner_drop_idx[1]][:3]

                pick1_xz = np.array([pick1[0]] + [pick1[2]])
                pick2_xz = np.array([pick2[0]] + [pick2[2]])

                end1_xz = np.array([end1[0]] + [end1[2]])
                end2_xz = np.array([end2[0]] + [end2[2]])

                if np.linalg.norm(abs(pick1_xz - end1_xz)) < 0.01 or \
                    np.linalg.norm(abs(pick2_xz - end2_xz)) < 0.01:
                    # print(f'Switch to 4\nend1: {end1}\nend2: {end2}\npick1: {pick1}\npick2: {pick2}')
                    self.expert_state = 4

            # drop
            else:
                # end1 = self.corner_drop[0]
                # end2 = self.corner_drop[1]
                end1 = particle_pos[self.corner_drop_idx[0]][:3]
                end2 = particle_pos[self.corner_drop_idx[1]][:3]

            if self.time_step > self.horizon - 10:
                # print(f'No time left. Switch to 4\nend1: {end1}\nend2: {end2}\npick1: {pick1}\npick2: {pick2}')
                self.expert_state = 4

            # ----------------------------------------------------------------------
            p_to_e1 = end1 - pick1
            p_to_e2 = end2 - pick2

            if self.expert_state == 0:

                temp = p_to_e1/(np.linalg.norm(p_to_e1) + 1e-8)
                act1 = np.hstack([temp, 0.])

                temp = p_to_e2/(np.linalg.norm(p_to_e2) + 1e-8)
                act2 = np.hstack([temp, 0.])

                act1[1] = 0.0
                act2[1] = 0.0

            elif self.expert_state == 1:
                temp = p_to_e1/(np.linalg.norm(p_to_e1) + 1e-8)
                act1 = np.hstack([temp, 0.0])

                temp = p_to_e2/(np.linalg.norm(p_to_e2) + 1e-8)
                act2 = np.hstack([temp, 0.0])

                act1[0] = 0.0
                act1[2] = 0.0

                act2[0] = 0.0
                act2[2] = 0.0

            elif self.expert_state == 2:
                temp = p_to_e1/(np.linalg.norm(p_to_e1) + 1e-8)
                act1 = np.hstack([temp, 1.])

                temp = p_to_e2/(np.linalg.norm(p_to_e2) + 1e-8)
                act2 = np.hstack([temp, 1.])

                act1[0] = 0.0
                act1[2] = 0.0

                act2[0] = 0.0
                act2[2] = 0.0

            elif self.expert_state == 3:
                temp = p_to_e1/(2*np.linalg.norm(p_to_e1) + 1e-8) # half of unit vector
                act1 = np.hstack([temp, [1.]])

                temp = p_to_e2/(2*np.linalg.norm(p_to_e2) + 1e-8) # half of unit vector
                act2 = np.hstack([temp, [1.]])

                act1[1] = 0.0
                act2[1] = 0.0

            else:
                act1 = np.hstack([p_to_e1, [0.]])
                act2 = np.hstack([p_to_e2, [0.]])
                act1[1] = 0.0
                act2[1] = 0.0

            # Combine
            expert_action = np.hstack([act1, act2])
        return expert_action
