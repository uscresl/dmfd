import numpy as np
from gym.spaces import Box
import pyflex
from softgym.envs.flex_env import FlexEnv
from softgym.action_space.action_space import  Picker, PickerPickPlace, PickerQPG, PickerPickAndPlace
from softgym.action_space.robot_env import RobotBase
from copy import deepcopy
from gym import spaces

class ClothEnv(FlexEnv):
    def __init__(self, observation_mode, action_mode, num_picker=2, render_mode='particle', picker_radius=0.05, picker_threshold=0.005, particle_radius=0.00625, **kwargs):
        self.render_mode = render_mode
        self.action_mode = action_mode
        self.cloth_particle_radius = particle_radius
        super().__init__(**kwargs)

        assert observation_mode in ['key_point', 'point_cloud', 'cam_rgb', 'cam_rgb_key_point', 'depth_key_point']
        assert action_mode in ['picker', 'pickerpickplace', 'pickerpickandplace', 'sawyer', 'franka', 'picker_qpg']
        self.observation_mode = observation_mode
        self.env_image_size = kwargs.get('env_image_size', 32)

        if action_mode == 'picker':
            if 'use_larger_action_space' in kwargs:
                picker_low = (-1.0, 0., -1.0)
                picker_high = (2.0, 0.5, 1.0)
                
            else:
                picker_low = (-0.4, 0., -0.4)
                picker_high = (1.0, 0.5, 0.4)
            self.action_tool = Picker(num_picker, picker_radius=picker_radius, particle_radius=particle_radius, picker_threshold=picker_threshold,
                                    picker_low=picker_low, picker_high=picker_high)
            self.action_space = self.action_tool.action_space
            self.picker_radius = picker_radius
        elif action_mode == 'pickerpickplace':
            self.action_tool = PickerPickPlace(num_picker=num_picker, particle_radius=particle_radius, env=self, picker_threshold=picker_threshold,
                                               picker_low=(-0.5, 0., -0.5), picker_high=(0.5, 0.3, 0.5))
            self.action_space = self.action_tool.action_space
            assert self.action_repeat == 1
        elif action_mode == 'pickerpickandplace':
            if self.__class__.__name__ == 'ClothFoldEnv':
                picker_low = (-0.9, 0., -0.9)
                picker_high = (0.9, 0.7, 0.9)
            elif self.__class__.__name__== 'DryClothEnv':
                picker_low = (-0.5, 0., -0.5)
                picker_high = (0.5, 0.7, 0.5)
            elif self.__class__.__name__== 'ClothFoldRobotHardEnv':
                picker_low = (-0.9, 0., -0.9)
                picker_high = (0.9, 0.7, 0.9)
            else:
                raise NotImplementedError
            self.action_tool = PickerPickAndPlace(num_picker=num_picker, particle_radius=particle_radius, env=self,
                                               picker_low=picker_low, picker_high=picker_high)
            self.action_space = self.action_tool.action_space
            assert self.action_repeat == 1
        elif action_mode in ['sawyer', 'franka']:
            self.action_tool = RobotBase(action_mode)
            self.action_space = self.action_tool.action_space
        elif action_mode == 'picker_qpg':
            cam_pos, cam_angle = self.get_camera_params()
            self.action_tool = PickerQPG((self.camera_height, self.camera_height), cam_pos, cam_angle, picker_threshold=picker_threshold,
                                         num_picker=num_picker, particle_radius=particle_radius, env=self,
                                         picker_low=(-0.3, 0., -0.3), picker_high=(0.3, 0.3, 0.3)
                                         )
            self.action_space = self.action_tool.action_space
        if observation_mode in ['key_point', 'point_cloud']:
            if observation_mode == 'key_point':
                obs_dim = len(self._get_key_point_idx()) * 3
            else:
                max_particles = 120 * 120
                obs_dim = max_particles * 3
                self.particle_obs_dim = obs_dim
            if action_mode.startswith('picker'):
                obs_dim += num_picker * 3
            else:
                raise NotImplementedError
            self.observation_space = Box(np.array([-np.inf] * obs_dim), np.array([np.inf] * obs_dim), dtype=np.float32)
        elif observation_mode == 'cam_rgb':
            self.observation_space = Box(low=-np.inf, high=np.inf, shape=(self.camera_height, self.camera_width, 3),
                                         dtype=np.float32)
        elif observation_mode == 'cam_rgb_key_point':
            obs_dim = len(self._get_key_point_idx()) * 3
            if action_mode.startswith('picker'):
                obs_dim += num_picker * 3
            else:
                raise NotImplementedError
            self.observation_space = spaces.Dict(
                dict(
                    image = Box(low=-np.inf, high=np.inf, shape=(self.env_image_size, self.env_image_size, 3), dtype=np.float32),
                    key_point = Box(np.array([-np.inf] * obs_dim), np.array([np.inf] * obs_dim), dtype=np.float32)
                )
            )
        elif observation_mode == 'depth_key_point':
            obs_dim = len(self._get_key_point_idx()) * 3
            if action_mode.startswith('picker'):
                obs_dim += num_picker * 3
            else:
                raise NotImplementedError
            self.observation_space = spaces.Dict(
                dict(
                    image = Box(low=-np.inf, high=np.inf, shape=(self.env_image_size, self.env_image_size, 1), dtype=np.float32),
                    key_point = Box(np.array([-np.inf] * obs_dim), np.array([np.inf] * obs_dim), dtype=np.float32)
                )
            )

    def _sample_cloth_size(self):
        return np.random.randint(60, 120), np.random.randint(60, 120)

    def _get_flat_pos(self):
        config = self.get_current_config()
        dimx, dimy = config['ClothSize']

        x = np.array([i * self.cloth_particle_radius for i in range(dimx)])
        y = np.array([i * self.cloth_particle_radius for i in range(dimy)])
        x = x - np.mean(x)
        y = y - np.mean(y)
        xx, yy = np.meshgrid(x, y)

        curr_pos = np.zeros([dimx * dimy, 3], dtype=np.float32)
        curr_pos[:, 0] = xx.flatten()
        curr_pos[:, 2] = yy.flatten()
        curr_pos[:, 1] = 5e-3  # Set specifally for particle radius of 0.00625
        return curr_pos

    def _set_to_flat(self):
        curr_pos = pyflex.get_positions().reshape((-1, 4))
        flat_pos = self._get_flat_pos()
        curr_pos[:, :3] = flat_pos
        pyflex.set_positions(curr_pos)
        pyflex.step()

    def get_camera_params(self):
        config = self.get_current_config()
        camera_name = config['camera_name']
        cam_pos = config['camera_params'][camera_name]['pos']
        cam_angle = config['camera_params'][camera_name]['angle']
        return cam_pos, cam_angle

    def get_default_config(self):
        """ Set the default config of the environment and load it to self.config """
        particle_radius = self.cloth_particle_radius
        if self.action_mode in ['sawyer', 'franka']:
            cam_pos, cam_angle = np.array([0.0, 1.62576, 1.04091]), np.array([0.0, -0.844739, 0])
        else:
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
            'flip_mesh': 0
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
                pos = np.concatenate([pos.flatten(), shapes[:, 0:3].flatten()])
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

    # Cloth index looks like the following:
    # 0, 1, ..., cloth_xdim -1
    # ...
    # cloth_xdim * (cloth_ydim -1 ), ..., cloth_xdim * cloth_ydim -1

    def _get_key_point_idx(self):
        """ The keypoints are defined as the four corner points of the cloth """
        dimx, dimy = self.current_config['ClothSize']
        idx_p1 = 0
        idx_p2 = dimx * (dimy - 1)
        idx_p3 = dimx - 1
        idx_p4 = dimx * dimy - 1
        return np.array([idx_p1, idx_p2, idx_p3, idx_p4])

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

    def normalize_action(self, action):
        lb, ub = self.action_space.low, self.action_space.high
        act_normalized = (action - lb) / (ub - lb) * 2 - 1 # normalize to range [-1, 1]
        return act_normalized

    def unnormalize_action(self, act_normalized):
        lb, ub = self.action_space.low, self.action_space.high
        act_unnormalized = ((act_normalized + 1) * (ub - lb)) / 2 + lb
        return act_unnormalized

    def normalize_action_pick_place(self, pick, place=np.array([0., 0., 0.])):
        assert pick.shape == (3,) # x,y,z location of pick
        scaled_action = np.concatenate([pick, place])
        lb, ub = self.action_space.low, self.action_space.high
        act_normalized = (scaled_action - lb) / (ub - lb) * 2 - 1 # normalize to range [-1, 1]
        return act_normalized

    def sim_to_real_action_mapping(self, act_normalized, angle):
        """
        Coordinate transforms from
        1. normalized policy actions to unnormalized actions (sim coordinates)
        2. sim coordinates to real coordinates
        """
        unact = self.unnormalize_action(act_normalized)

        # reoder them so they are in pick xzy place xzy
        sim_pick = np.array([unact[0], unact[2], unact[1]])
        sim_place = np.array([unact[3], unact[5], unact[4]])

        ### ALL XYZ in REAL ROBOT FRAME
        sim_min = np.array([-0.47, 0.47, 0.]) # IN REAL ROBOT FRAME
        sim_max = np.array([0.47, -0.47, 1.]) # IN REAL ROBOT FRAME

        # Whiteboard setup 5/4
        # real_min = np.array([0.7252, -0.251, 0.030792])
        # real_max = np.array([0.2804, 0.207, 0.04])

        # Whiteboard setup 5/6
        real_min = np.array([0.739652, -0.278651, 0.030792])
        real_max = np.array([0.292201, 0.211139 , 0.04])

        def map_sim_to_real(sim_xyz):
            real_xyz = np.zeros(3) # fill this up

            xf = (sim_xyz[0] - sim_min[0])/(sim_max[0] - sim_min[0])
            yf = (sim_xyz[1] - sim_min[1])/(sim_max[1] - sim_min[1])
            zf = (sim_xyz[2] - sim_min[2])/(sim_max[2] - sim_min[2])

            real_xyz[0] = real_min[0] + xf * (real_max[0] - real_min[0])
            real_xyz[1] = real_min[1] + yf * (real_max[1] - real_min[1])
            real_xyz[2] = real_min[2] + zf * (real_max[2] - real_min[2])

            return real_xyz

        real_pick = map_sim_to_real(sim_pick)
        real_place = map_sim_to_real(sim_place)

        x_offsets = 0.06
        z_offsets = 0.06

        real_pick_adjusted = real_pick.copy()
        real_place_adusted = real_place.copy()

        real_pick_adjusted[0] -= x_offsets
        real_place_adusted[0] += x_offsets

        real_pick_adjusted[1] += z_offsets
        real_place_adusted[1] -= z_offsets

        print('Real pick adjusted (without factoring in angle): ', real_pick_adjusted)
        print('Real place adjusted (without factoring in angle): ', real_place_adusted)

        if abs(angle) > 16:
            # 150 would scale up to quickly as angle gets bigger
            angle_offset_pick_factor = 200
            angle_offset_place_factor = 150
        else:
            angle_offset_pick_factor = 150
            angle_offset_place_factor = 150

        if angle < 0:
            # cloth has been left rotated
            angle = abs(angle)
            angle_offset_pick = angle / angle_offset_pick_factor
            angle_offset_place = angle / angle_offset_place_factor

            # pick should move more inward (left to right)
            # place should move more inward (right to left)
            real_pick_adjusted[0] += angle_offset_pick
            real_place_adusted[0] -= angle_offset_place

            real_pick_adjusted[1] += angle_offset_pick
            real_place_adusted[1] -= angle_offset_place
        else:
            angle_offset_pick = angle / angle_offset_pick_factor
            angle_offset_place = angle / angle_offset_place_factor

            # pick should move more outward (right to left)
            # place should move more outward (left to right)
            real_pick_adjusted[0] -= angle_offset_pick
            real_place_adusted[0] += angle_offset_place

            real_pick_adjusted[1] -= angle_offset_pick
            real_place_adusted[1] += angle_offset_place
        return real_pick, real_place, real_pick_adjusted, real_place_adusted
