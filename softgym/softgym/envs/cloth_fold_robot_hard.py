import numpy as np
import pyflex
from copy import deepcopy
from softgym.envs.cloth_env import ClothEnv
from softgym.utils.pyflex_utils import center_object
from softgym.utils.misc import quatFromAxisAngle
from softgym.action_space.action_space import  Picker

class ClothFoldRobotHardEnv(ClothEnv):
    '''
    This is used for sim-to-real experiments.
    '''
    def __init__(self, cached_states_path='cloth_fold_robot_hard_init_states.pkl', **kwargs):
        self.bottom_left_corner_pos = self.top_right_corner_pos = self.top_right_corner_box_height = None

        self.default_cloth_height = 0.1
        if kwargs['action_mode'] == 'pickerpickandplace':
          self.default_action_tool_height = 0.2
        else:
          self.default_action_tool_height = 0.18
          kwargs['use_larger_action_space'] = True
        self.default_table_height = 0.1
        self.default_box_height = 0.04

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

    def get_default_config(self):
        """ Set the default config of the environment and load it to self.config """
        particle_radius = self.cloth_particle_radius
        if self.action_mode in ['sawyer', 'franka']:
            cam_pos, cam_angle = np.array([0.0, 1.62576, 1.04091]), np.array([0.0, -0.844739, 0])
        else:
          if self.action_mode == 'pickerpickandplace':
            cam_pos, cam_angle = np.array([-0.0, 1.4, 0]), np.array([0, -90 / 180. * np.pi, 0.])
          else:
            # cam_pos, cam_angle = np.array([-0.0, 2, 2]), np.array([0, -45 / 180. * np.pi, 0.]) # old side view
            cam_pos, cam_angle = np.array([-0.0, 1.5, 0.4]), np.array([0, -75 / 180. * np.pi, 0.])
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

    def _get_obs(self):
        if self.observation_mode in ['cam_rgb_key_point', 'depth_key_point']:
            particle_pos = np.array(pyflex.get_positions()).reshape([-1, 4])[:, :3]
            keypoint_pos = particle_pos[self._get_key_point_idx(), :3]
            pos = keypoint_pos

            if self.action_mode in ['sphere', 'picker']:
                shapes = pyflex.get_shape_states()
                shapes = np.reshape(shapes, [-1, 14])
                # only one actuator (the other two are table and box)
                pos = np.concatenate([pos.flatten(), shapes[0, 0:3].flatten()])
            elif self.action_mode == 'pickerpickandplace':
                shapes = pyflex.get_shape_states()
                shapes = np.reshape(shapes, [-1, 14])
                # extract actuators' shape and exclude rack in DryCloth
                pos = np.concatenate([pos.flatten(), shapes[:self.action_tool.num_picker, 0:3].flatten()])

            if self.observation_mode == 'cam_rgb_key_point':
                pos = {
                    'image': self.get_image(self.env_image_size, self.env_image_size),
                    'key_point': pos,
                }
            elif self.observation_mode == 'depth_key_point':
                pos = {
                    'image': self.get_depth_image(self.env_image_size, self.env_image_size),
                    'key_point': pos,
                }
            else:
                raise NotImplementedError
        else:
            if self.observation_mode == 'cam_rgb':
                return self.get_image(self.camera_height, self.camera_width)
            if self.observation_mode == 'point_cloud':
                particle_pos = np.array(pyflex.get_positions()).reshape([-1, 4])[:, :3].flatten()
                pos = np.zeros(shape=self.particle_obs_dim, dtype=np.float)
                pos[:len(particle_pos)] = particle_pos
            elif self.observation_mode == 'key_point':
                particle_pos = np.array(pyflex.get_positions()).reshape([-1, 4])[:, :3]
                keypoint_pos = particle_pos[self._get_key_point_idx(), :3]
                pos = keypoint_pos

            if self.action_mode in ['sphere', 'picker']:
                shapes = pyflex.get_shape_states()
                shapes = np.reshape(shapes, [-1, 14])
                pos = np.concatenate([pos.flatten(), shapes[:, 0:3].flatten()])
            elif self.action_mode == 'pickerpickandplace':
                shapes = pyflex.get_shape_states()
                shapes = np.reshape(shapes, [-1, 14])
                # extract actuators' shape and exclude rack in DryCloth
                pos = np.concatenate([pos.flatten(), shapes[:self.action_tool.num_picker, 0:3].flatten()])
        return pos

    def set_scene(self, config, state=None):
        if self.render_mode == 'particle':
            render_mode = 1
        elif self.render_mode == 'cloth':
            render_mode = 2
        elif self.render_mode == 'both':
            render_mode = 3
        camera_params = config['camera_params'][config['camera_name']]
        env_idx = 0 if 'env_idx' not in config else config['env_idx']
        mass = config['mass'] if 'mass' in config else 0.5
        scene_params = np.array([*config['ClothPos'], *config['ClothSize'], *config['ClothStiff'], render_mode,
                                 *camera_params['pos'][:], *camera_params['angle'][:], camera_params['width'], camera_params['height'], mass,
                                 config['flip_mesh']])

        if self.version == 2:
            robot_params = [1.] if self.action_mode in ['sawyer', 'franka'] else []
            self.params = (scene_params, robot_params)
            pyflex.set_scene(env_idx, scene_params, 0, robot_params)
        elif self.version == 1:
            pyflex.set_scene(env_idx, scene_params, 0)

        if state is not None:
            self.set_state(state)
        self.current_config = deepcopy(config)

    def randomize_xz_pos(self):
        pos = pyflex.get_positions().reshape(-1, 4)

        random_candidates = [round(i, 2) for i in np.arange(-0.1, 0.11, 0.02)]

        x_offset = np.random.choice(random_candidates)
        z_offset = np.random.choice(random_candidates)

        # between -0.1 and 0.1
        pos[:, 0] += x_offset
        pos[:, 2] += z_offset

        pyflex.set_positions(pos)

        return x_offset, z_offset

    def rotate_particles(self, angle):
        pos = pyflex.get_positions().reshape(-1, 4)
        center = np.mean(pos, axis=0)
        pos -= center
        new_pos = pos.copy()
        new_pos[:, 0] = (np.cos(angle) * pos[:, 0] - np.sin(angle) * pos[:, 2])
        new_pos[:, 2] = (np.sin(angle) * pos[:, 0] + np.cos(angle) * pos[:, 2])
        new_pos += center
        pyflex.set_positions(new_pos)

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

            # add variations to xz
            x_offset, z_offset = self.randomize_xz_pos()

            # add rotations variations - max 45 degrees
            angle = (np.random.random() - 0.5) * np.pi / 8
            self.rotate_particles(angle)

            generated_configs.append(deepcopy(config))
            print('config {}: {}, x_offset {} z_offset {} angle {}'.format(i, config['camera_params'], x_offset, z_offset, angle))
            generated_states.append(deepcopy(self.get_state()))

        return generated_configs, generated_states

    def reset(self, config=None, initial_state=None, config_id=None, is_eval=False):
        if config is None:
            if config_id is None:
                if self.eval_flag:
                    eval_beg = int(0.8 * len(self.cached_configs))
                    config_id = np.random.randint(low=eval_beg, high=len(self.cached_configs)) if not self.deterministic else eval_beg
                else:
                     # Only 800 variations will be used for training. Since we're using this for sim-to-real, we want as many variations as possible.
                    # train_high = int(0.8 * len(self.cached_configs))
                    train_high = len(self.cached_configs)
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

            # set it to 0.15 to account for the table. Table's height is 0.1.
            self.action_tool.reset([middle_point[0], self.default_action_tool_height, middle_point[2]])

        config = self.get_current_config()
        num_particles = np.prod(config['ClothSize'], dtype=int)
        particle_grid_idx = np.array(list(range(num_particles))).reshape(config['ClothSize'][1], config['ClothSize'][0])  # Reversed index here

        cloth_dimx = config['ClothSize'][0]

        self.bottom_left_corner_pos = particle_grid_idx[-1, 0]
        self.top_right_corner_pos = particle_grid_idx[0, -1]
        self.top_right_corner_box_height = 0.14

        # Edge particles
        self.top_edge_indices = particle_grid_idx[0, :]
        self.bottom_edge_indices = particle_grid_idx[-1, :]
        self.left_edge_indices = particle_grid_idx[:, 0]
        self.right_edge_indices = particle_grid_idx[:, -1]

        # reset cloth height
        pos = pyflex.get_positions().reshape(-1, 4)
        pos[:, 1] = self.default_cloth_height
        pyflex.set_positions(pos.flatten())

        # table under the cloth
        halfEdge = np.array([0.8, self.default_table_height, 0.8])
        center = np.array([0, 0, 0])
        quat = quatFromAxisAngle([0, 0, -1.], 0.)
        pyflex.add_box(halfEdge, center, quat)
        pyflex.step()

        self.performance_corner_init = None
        self.performance_edges_init = None
        self.performance_init = self.performance_edges_init
        info = self._get_info()
        self.performance_corner_init = info['performance_corner']
        self.performance_edges_init = info['performance_edges']
        self.performance_init = self.performance_edges_init

        if self.action_mode == 'picker':
            self.top_right_idx = particle_grid_idx[0, -1]
            self.bottom_left_idx = particle_grid_idx[-1, 0]
            self.expert_state = 0
            self.prev_distance = -np.inf
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
        # print(f'Step {self.time_step}, norm_perf_corner {info["normalized_performance_corner"]:.4f}, norm_perf_edges {info["normalized_performance_edges"]:.4f}, norm_perf {info["normalized_performance"]:.4f}')

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

    def _sample_cloth_size(self):
        return 120, 120

    def _get_obs_key_points(self):
        particle_pos = np.array(pyflex.get_positions()).reshape([-1, 4])[:, :3]
        keypoint_pos = particle_pos[self._get_key_point_idx(), :3]
        pos = keypoint_pos

        if self.action_mode in ['sphere', 'picker', 'pickerpickandplace']:
            shapes = pyflex.get_shape_states()
            shapes = np.reshape(shapes, [-1, 14])
            # only one actuator (the other two are table and box)
            pos = np.concatenate([pos.flatten(), shapes[0, 0:3].flatten()])
        return pos

    def compute_reward(self, action=None, obs=None, set_prev_reward=False, expert_action=None, rew_info=False):
        """
        The reward is the negative difference betweeen the bottom left corner of the
        cloth to the top right corner of the box.
        """
        pos = pyflex.get_positions()
        pos = pos.reshape((-1, 4))[:, :3]

        pos_bottom_left_corner_cloth = pos[self.bottom_left_corner_pos]
        pos_top_right_corner_box = pos[self.top_right_corner_pos]
        pos_top_right_corner_box[1] = self.top_right_corner_box_height

        pos_left_edge = pos[self.left_edge_indices]
        pos_right_edge = pos[self.right_edge_indices]
        pos_top_edge = pos[self.top_edge_indices]
        pos_bottom_edge = pos[self.bottom_edge_indices]

        curr_dist = np.mean(np.linalg.norm(pos_bottom_left_corner_cloth - pos_top_right_corner_box))
        # reward = -curr_dist

        dist_edges = np.linalg.norm(pos_left_edge - pos_top_edge) + np.linalg.norm(pos_right_edge - pos_bottom_edge)
        reward = -dist_edges

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
            return reward, {'il_reward': il_reward, 'task_reward': -dist_edges, 'task_reward_corner': -curr_dist, 'task_reward_edges': -dist_edges}

        return reward

    def _get_info(self):
        # Duplicate of the compute reward function!
        pos = pyflex.get_positions()
        pos = pos.reshape((-1, 4))[:, :3]
        pos_bottom_left_corner_cloth = pos[self.bottom_left_corner_pos]
        pos_top_right_corner_box = pos[self.top_right_corner_pos]
        pos_top_right_corner_box[1] = self.top_right_corner_box_height

        pos_left_edge = pos[self.left_edge_indices]
        pos_right_edge = pos[self.right_edge_indices]
        pos_top_edge = pos[self.top_edge_indices]
        pos_bottom_edge = pos[self.bottom_edge_indices]

        dist = np.mean(np.linalg.norm(pos_bottom_left_corner_cloth - pos_top_right_corner_box))
        performance_corner = -dist
        performance_corner_init = performance_corner if self.performance_corner_init is None else self.performance_corner_init  # Use the original performance

        dist_edges = np.linalg.norm(pos_left_edge - pos_top_edge) + np.linalg.norm(pos_right_edge - pos_bottom_edge)
        performance_edges = -dist_edges
        performance_edges_init = performance_edges if self.performance_edges_init is None else self.performance_edges_init

        info = {
            'performance_corner': performance_corner,
            'normalized_performance_corner': (performance_corner - performance_corner_init) / (0. - performance_corner_init),
            'performance': performance_edges,
            'normalized_performance': (performance_edges - performance_edges_init) / (0. - performance_edges_init),
            'performance_edges': performance_edges,
            'normalized_performance_edges': (performance_edges - performance_edges_init) / (0. - performance_edges_init),
        }
        if 'qpg' in self.action_mode:
            info['total_steps'] = self.action_tool.total_steps
        return info

    def compute_expert_action(self):
      if self.action_mode == 'pickerpickandplace':
        particle_pos = pyflex.get_positions().reshape(-1, 4)
        pick = particle_pos[self.bottom_left_corner_pos][:3]
        pick[1] = 0.1

        place = particle_pos[self.top_right_corner_pos][:3]
        place[1] = 0.2
        place[0] += 0.06 # move a littler closter to the top right corner
        place[2] -= 0.06 # move a littler closter to the top right corner
        action = np.concatenate((pick, place))
        action = self.normalize_action(action)
      else:
        picker_pos, _ = self.action_tool._get_pos()
        particle_pos = pyflex.get_positions().reshape((-1, 4))

        pick = picker_pos[0, :3]
        end = None
        not_moving_thresh = 1e-7

        do_pick_thresh = self.action_tool.picker_radius + self.action_tool.particle_radius + self.action_tool.picker_threshold

        # go to the corner
        if self.expert_state == 0:
            end = particle_pos[self.bottom_left_idx][:3]
            pick_xz = np.array([pick[0]] + [pick[2]])
            end_xz = np.array([end[0]] + [end[2]])

            if abs(np.linalg.norm(abs(pick_xz - end_xz)) -  self.prev_distance) < not_moving_thresh:
                self.expert_state = 1
                self.prev_distance = -np.inf
            else:
                self.prev_distance = abs(np.linalg.norm(abs(pick_xz - end_xz)))
        # go down to the corner
        elif self.expert_state == 1:
            end = particle_pos[self.bottom_left_idx][:3]

            if np.linalg.norm(abs(pick - end)) < do_pick_thresh:
                self.expert_state = 2
        # pick up
        elif self.expert_state == 2:
            end = particle_pos[self.bottom_left_idx][:3]
            end[1] = 0.23

            if np.linalg.norm(abs(pick - end)) < do_pick_thresh:
                self.expert_state = 3
        # go to the goal corner
        elif self.expert_state == 3:
            end = particle_pos[self.top_right_corner_pos][:3]

            pick_xz = np.array([pick[0]] + [pick[2]])
            end_xz = np.array([end[0]] + [end[2]])

            if abs(np.linalg.norm(abs(pick_xz - end_xz)) -  self.prev_distance) < not_moving_thresh:
                self.expert_state = 4
                self.prev_distance = -np.inf
            else:
                self.prev_distance = abs(np.linalg.norm(abs(pick_xz - end_xz)))

        if end is not None:
            p_to_e1 = (end - pick)

        if self.expert_state == 0:
            temp = p_to_e1/(np.linalg.norm(p_to_e1) + 1e-8)
            action = np.hstack([temp, 0.])
            action[1] = 0
        elif self.expert_state == 1:
            temp = p_to_e1/(np.linalg.norm(p_to_e1) + 1e-8)
            action = np.hstack([temp, 0.0])
            action[0] = 0.0
            action[2] = 0.0
        elif self.expert_state == 2:
            temp = p_to_e1/(np.linalg.norm(p_to_e1) + 1e-8)
            action = np.hstack([temp, 1.])
            action[0] = 0.0
        elif self.expert_state == 3:
            temp = p_to_e1/(np.linalg.norm(p_to_e1) + 1e-8)
            action = np.hstack([temp, [1.]])
            action[1] = 0.0
        else:
            action = np.array([0, 0, 0, 0])

      return action
