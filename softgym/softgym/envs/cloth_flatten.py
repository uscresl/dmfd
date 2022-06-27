import numpy as np
import random
import pyflex
from softgym.envs.cloth_env import ClothEnv
from copy import deepcopy
from softgym.utils.misc import vectorized_range, vectorized_meshgrid
from softgym.utils.pyflex_utils import center_object


class ClothFlattenEnv(ClothEnv):
    def __init__(self, cached_states_path='cloth_flatten_init_states.pkl', **kwargs):
        """
        :param cached_states_path:
        :param num_picker: Number of pickers if the aciton_mode is picker
        :param kwargs:
        """
        super().__init__(**kwargs)
        self.get_cached_configs_and_states(cached_states_path, self.num_variations)
        self.prev_covered_area = None  # Should not be used until initialized

        # expert policy
        self.ep_move_xy_timesteps_thresh = 2

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

    def generate_env_variation(self, num_variations=1, vary_cloth_size=True):
        """ Generate initial states. Note: This will also change the current states! """
        max_wait_step = 300  # Maximum number of steps waiting for the cloth to stablize
        stable_vel_threshold = 0.01  # Cloth stable when all particles' vel are smaller than this
        generated_configs, generated_states = [], []
        default_config = self.get_default_config()

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
            if self.action_mode in ['sawyer', 'franka']:  # Take care of the table in robot case
                pos[:, 1] = 0.57
            else:
                pos[:, 1] = 0.005
            pos[:, 3] = 1
            pyflex.set_positions(pos.flatten())
            pyflex.set_velocities(np.zeros_like(pos))
            pyflex.step()

            num_particle = cloth_dimx * cloth_dimy
            pickpoint = random.randint(0, num_particle - 1)
            curr_pos = pyflex.get_positions()
            original_inv_mass = curr_pos[pickpoint * 4 + 3]
            curr_pos[pickpoint * 4 + 3] = 0  # Set the mass of the pickup point to infinity so that it generates enough force to the rest of the cloth
            pickpoint_pos = curr_pos[pickpoint * 4: pickpoint * 4 + 3].copy()  # Pos of the pickup point is fixed to this point
            pickpoint_pos[1] += np.random.random(1) * 0.5 + 0.5
            pyflex.set_positions(curr_pos)

            # Pick up the cloth and wait to stablize
            for j in range(0, max_wait_step):
                curr_pos = pyflex.get_positions()
                curr_vel = pyflex.get_velocities()
                curr_pos[pickpoint * 4: pickpoint * 4 + 3] = pickpoint_pos
                curr_vel[pickpoint * 3: pickpoint * 3 + 3] = [0, 0, 0]
                pyflex.set_positions(curr_pos)
                pyflex.set_velocities(curr_vel)
                pyflex.step()
                if np.alltrue(np.abs(curr_vel) < stable_vel_threshold) and j > 5:
                    break

            # Drop the cloth and wait to stablize
            curr_pos = pyflex.get_positions()
            curr_pos[pickpoint * 4 + 3] = original_inv_mass
            pyflex.set_positions(curr_pos)
            for _ in range(max_wait_step):
                pyflex.step()
                curr_vel = pyflex.get_velocities()
                if np.alltrue(curr_vel < stable_vel_threshold):
                    break

            center_object()

            if self.action_mode == 'sphere' or self.action_mode.startswith('picker'):
                curr_pos = pyflex.get_positions()
                self.action_tool.reset(curr_pos[pickpoint * 4:pickpoint * 4 + 3] + [0., 0.2, 0.])
            generated_configs.append(deepcopy(config))
            generated_states.append(deepcopy(self.get_state()))
            self.current_config = config  # Needed in _set_to_flatten function
            generated_configs[-1]['flatten_area'] = self._set_to_flatten()  # Record the maximum flatten area

            print('config {}: camera params {}, flatten area: {}'.format(i, config['camera_params'], generated_configs[-1]['flatten_area']))

        return generated_configs, generated_states

    def _set_to_flatten(self):
        # self._get_current_covered_area(pyflex.get_positions().reshape(-))
        cloth_dimx, cloth_dimz = self.get_current_config()['ClothSize']
        N = cloth_dimx * cloth_dimz
        px = np.linspace(0, cloth_dimx * self.cloth_particle_radius, cloth_dimx)
        py = np.linspace(0, cloth_dimz * self.cloth_particle_radius, cloth_dimz)
        xx, yy = np.meshgrid(px, py)
        new_pos = np.empty(shape=(N, 4), dtype=np.float)
        new_pos[:, 0] = xx.flatten()
        new_pos[:, 1] = self.cloth_particle_radius
        new_pos[:, 2] = yy.flatten()
        new_pos[:, 3] = 1.
        new_pos[:, :3] -= np.mean(new_pos[:, :3], axis=0)
        pyflex.set_positions(new_pos.flatten())
        return self._get_current_covered_area(new_pos)


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
        """ Right now only use one initial state"""
        self.prev_covered_area = self._get_current_covered_area(pyflex.get_positions())
        if hasattr(self, 'action_tool'):
            curr_pos = pyflex.get_positions()
            cx, cy = self._get_center_point(curr_pos)
            self.action_tool.reset([cx, 0.2, cy])
        pyflex.step()
        self.init_covered_area = None
        info = self._get_info()
        self.init_covered_area = info['performance']

        self.expert_state = 0
        self.corner_pick = []
        self.corner_pick_idx = []
        self.corner_drop = []
        self.past_dir = -1
        self.past_ee_loc = [[None], [None]]
        self.ep_move_xy_timesteps = 0

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
            pyflex.step(self.action_tool.next_action)
        else:
            pyflex.step()
        return

    def _get_current_covered_area(self, pos):
        """
        Calculate the covered area by taking max x,y cood and min x,y coord, create a discritized grid between the points
        :param pos: Current positions of the particle states
        """
        pos = np.reshape(pos, [-1, 4])
        min_x = np.min(pos[:, 0])
        min_y = np.min(pos[:, 2])
        max_x = np.max(pos[:, 0])
        max_y = np.max(pos[:, 2])
        init = np.array([min_x, min_y])
        span = np.array([max_x - min_x, max_y - min_y]) / 100.
        pos2d = pos[:, [0, 2]]

        offset = pos2d - init
        slotted_x_low = np.maximum(np.round((offset[:, 0] - self.cloth_particle_radius) / span[0]).astype(int), 0)
        slotted_x_high = np.minimum(np.round((offset[:, 0] + self.cloth_particle_radius) / span[0]).astype(int), 100)
        slotted_y_low = np.maximum(np.round((offset[:, 1] - self.cloth_particle_radius) / span[1]).astype(int), 0)
        slotted_y_high = np.minimum(np.round((offset[:, 1] + self.cloth_particle_radius) / span[1]).astype(int), 100)
        # Method 1
        grid = np.zeros(10000)  # Discretization
        listx = vectorized_range(slotted_x_low, slotted_x_high)
        listy = vectorized_range(slotted_y_low, slotted_y_high)
        listxx, listyy = vectorized_meshgrid(listx, listy)
        idx = listxx * 100 + listyy
        idx = np.clip(idx.flatten(), 0, 9999)
        grid[idx] = 1

        return np.sum(grid) * span[0] * span[1]

        # Method 2
        # grid_copy = np.zeros([100, 100])
        # for x_low, x_high, y_low, y_high in zip(slotted_x_low, slotted_x_high, slotted_y_low, slotted_y_high):
        #     grid_copy[x_low:x_high, y_low:y_high] = 1
        # assert np.allclose(grid_copy, grid)
        # return np.sum(grid_copy) * span[0] * span[1]

    def _get_center_point(self, pos):
        pos = np.reshape(pos, [-1, 4])
        min_x = np.min(pos[:, 0])
        min_y = np.min(pos[:, 2])
        max_x = np.max(pos[:, 0])
        max_y = np.max(pos[:, 2])
        return 0.5 * (min_x + max_x), 0.5 * (min_y + max_y)

    def _get_obs_key_points(self):
        particle_pos = np.array(pyflex.get_positions()).reshape([-1, 4])[:, :3]
        keypoint_pos = particle_pos[self._get_key_point_idx(), :3]
        pos = keypoint_pos

        if self.action_mode in ['sphere', 'picker']:
            shapes = pyflex.get_shape_states()
            shapes = np.reshape(shapes, [-1, 14])
            pos = np.concatenate([pos.flatten(), shapes[:, 0:3].flatten()])
        return pos

    def compute_reward(self, action=None, obs=None, set_prev_reward=False, expert_action=None, rew_info=False):
        particle_pos = pyflex.get_positions()
        curr_covered_area = self._get_current_covered_area(particle_pos)
        r = curr_covered_area

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
        r += il_reward
        if rew_info:
            return r, {'il_reward': il_reward, 'task_reward': curr_covered_area}


        return r

    # @property
    # def performance_bound(self):
    #     dimx, dimy = self.current_config['ClothSize']
    #     max_area = dimx * self.cloth_particle_radius * dimy * self.cloth_particle_radius
    #     min_p = 0
    #     max_p = max_area
    #     return min_p, max_p

    def _get_info(self):
        # Duplicate of the compute reward function!
        particle_pos = pyflex.get_positions()
        curr_covered_area = self._get_current_covered_area(particle_pos)
        init_covered_area = curr_covered_area if self.init_covered_area is None else self.init_covered_area
        max_covered_area = self.get_current_config()['flatten_area']
        info = {
            'performance': curr_covered_area,
            'normalized_performance': (curr_covered_area - init_covered_area) / (max_covered_area - init_covered_area),
        }
        if 'qpg' in self.action_mode:
            info['total_steps'] = self.action_tool.total_steps
        return info

    def get_picked_particle(self):
        pps = np.ones(shape=self.action_tool.num_picker)  * -1 # -1 means no particles picked
        for i, pp in enumerate(self.action_tool.picked_particles):
            if pp is not None:
                pps[i] = pp
        return pps

    def get_corners(self):

        particle_pos = pyflex.get_positions().reshape(-1, 4)

        self.x_min = np.amin(particle_pos[:, 0])
        self.x_max = np.amax(particle_pos[:, 0])

        self.z_min = np.amin(particle_pos[:, 2])
        self.z_max = np.amax(particle_pos[:, 2])

        # checked and saw that these were all the same
        # may not be the same in this case -------------------------------------
        #import pdb; pdb.set_trace()
        self.y_coor = particle_pos[0, 1]

        # corners 1,2 and 3,4 are opposite
        self.corner1_idx  = np.where(particle_pos[:, 0] == self.x_min)[0][0]
        self.corner2_idx  = np.where(particle_pos[:, 0] == self.x_max)[0][0]
        self.corner3_idx  = np.where(particle_pos[:, 2] == self.z_min)[0][0]
        self.corner4_idx  = np.where(particle_pos[:, 2] == self.z_max)[0][0]

        self.corner1 = particle_pos[self.corner1_idx][:3]
        self.corner2 = particle_pos[self.corner2_idx][:3]
        self.corner3 = particle_pos[self.corner3_idx][:3]
        self.corner4 = particle_pos[self.corner4_idx][:3]

    def compute_expert_action(self):
            """ Simple (suboptimal) expert: Pick two corners and move them apart
            """

            picker_pos, particle_pos = self.action_tool._get_pos()
            # locate corners of particle_pos and then do 4 picks instead of two
            pick1 = picker_pos[0, :3]
            pick2 = picker_pos[-1, :3]

            do_pick_thresh = self.action_tool.picker_radius + self.action_tool.particle_radius + self.action_tool.picker_threshold

            self.get_corners()

            # go to center between corner1 and corner2
            if self.expert_state == 0:
                center = (self.corner1 + self.corner2) / 2
                end1 = center
                end2 = center

                pick1_xz = np.array([pick1[0]] + [pick1[2]])
                pick2_xz = np.array([pick2[0]] + [pick2[2]])

                end1_xz = np.array([end1[0]] + [end1[2]])
                end2_xz = np.array([end2[0]] + [end2[2]])

                if np.linalg.norm(abs(pick1_xz - end1_xz)) < 0.01 and \
                    np.linalg.norm(abs(pick2_xz - end2_xz)) < 0.01:
                    self.expert_state = 1

            # move down to touch cloth
            elif self.expert_state == 1:
                center = (self.corner1 + self.corner2) / 2
                end1 = center
                end2 = center

                if np.linalg.norm(abs(pick1 - end1)) < do_pick_thresh or \
                    np.linalg.norm(abs(pick2 - end2)) < do_pick_thresh:
                    self.expert_state = 2

            # one eff goes to corner 1 and the other eff goes to corner 2
            elif self.expert_state == 2:
                end1 = self.corner1
                end2 = self.corner2

                pick1_xz = np.array([pick1[0]] + [pick1[2]])
                pick2_xz = np.array([pick2[0]] + [pick2[2]])

                end1_xz = np.array([end1[0]] + [end1[2]])
                end2_xz = np.array([end2[0]] + [end2[2]])

                self.ep_move_xy_timesteps += 1
                if self.ep_move_xy_timesteps > self.ep_move_xy_timesteps_thresh:
                    self.expert_state = 3
                    self.ep_move_xy_timesteps = 0

            # move effs up in the air
            elif self.expert_state == 3:
                # center = (self.corner3 + self.corner4) / 2
                end1 = self.corner1
                end2 = self.corner2

                end2[1] = 0.5
                end1[1] = 0.5

                pick1_xz = np.array([pick1[0]] + [pick1[2]])
                pick2_xz = np.array([pick2[0]] + [pick2[2]])

                end1_xz = np.array([end1[0]] + [end1[2]])
                end2_xz = np.array([end2[0]] + [end2[2]])

            # go to the center between corner3 and corner4
            elif self.expert_state == 4:
                center = (self.corner3 + self.corner4) / 2
                end1 = center
                end2 = center

                pick1_xz = np.array([pick1[0]] + [pick1[2]])
                pick2_xz = np.array([pick2[0]] + [pick2[2]])

                end1_xz = np.array([end1[0]] + [end1[2]])
                end2_xz = np.array([end2[0]] + [end2[2]])

                if np.linalg.norm(abs(pick1_xz - end1_xz)) < 0.01 and \
                    np.linalg.norm(abs(pick2_xz - end2_xz)) < 0.01:
                    self.expert_state = 5

            # move down to touch cloth
            elif self.expert_state == 5:
                center = (self.corner3 + self.corner4) / 2
                end1 = center
                end2 = center

                if np.linalg.norm(abs(pick1 - end1)) < do_pick_thresh or \
                    np.linalg.norm(abs(pick2 - end2)) < do_pick_thresh:
                    self.expert_state = 6

            # one eff goes to corner 3 and the other eff goes to corner 4
            elif self.expert_state == 6:
                end1 = self.corner3
                end2 = self.corner4

                pick1_xz = np.array([pick1[0]] + [pick1[2]])
                pick2_xz = np.array([pick2[0]] + [pick2[2]])

                end1_xz = np.array([end1[0]] + [end1[2]])
                end2_xz = np.array([end2[0]] + [end2[2]])

                self.ep_move_xy_timesteps += 1
                if self.ep_move_xy_timesteps > self.ep_move_xy_timesteps_thresh:
                    self.expert_state = 7
                    self.ep_move_xy_timesteps = 0

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
                act1 = np.hstack([temp, 0.])

                temp = p_to_e2/(np.linalg.norm(p_to_e2) + 1e-8)
                act2 = np.hstack([temp, 0.])

                act1[1] = 0.0
                act2[1] = 0.0

            elif self.expert_state == 3:
                act1 = np.hstack([p_to_e1, [0.]])
                act2 = np.hstack([p_to_e2, [0.]])
                act1[1] = 1.0
                act2[1] = 1.0
                self.expert_state = 4

            elif self.expert_state == 4:
                temp = p_to_e1/(np.linalg.norm(p_to_e1) + 1e-8)
                act1 = np.hstack([temp, 0.])

                temp = p_to_e2/(np.linalg.norm(p_to_e2) + 1e-8)
                act2 = np.hstack([temp, 0.])

                act1[1] = 0.0
                act2[1] = 0.0
            elif self.expert_state == 5:
                temp = p_to_e1/(np.linalg.norm(p_to_e1) + 1e-8)
                act1 = np.hstack([temp, 0.0])

                temp = p_to_e2/(np.linalg.norm(p_to_e2) + 1e-8)
                act2 = np.hstack([temp, 0.0])

                act1[0] = 0.0
                act1[2] = 0.0

                act2[0] = 0.0
                act2[2] = 0.0

            elif self.expert_state == 6:
                temp = p_to_e1/(np.linalg.norm(p_to_e1) + 1e-8)
                act1 = np.hstack([temp, 0.])

                temp = p_to_e2/(np.linalg.norm(p_to_e2) + 1e-8)
                act2 = np.hstack([temp, 0.])

                act1[1] = 0.0
                act2[1] = 0.0
            elif self.expert_state == 7:
                act1 = np.hstack([p_to_e1, [0.]])
                act2 = np.hstack([p_to_e2, [0.]])
                act1[1] = 1.0
                act2[1] = 1.0
                self.expert_state = 0

            expert_action = np.hstack([act1, act2])

            return expert_action
