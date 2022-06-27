import argparse
import numpy as np
import torch
import random
import os
import json

# For CLI arguments with argparse
def str2bool(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, str) and v.lower() in ('true', ):
        return True
    elif isinstance(v, str) and v.lower() in ('false', ):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected')

def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def center_crop_image(image, output_size):
    # print('input image shape:', image.shape)
    if image.shape[0] ==1:
        image = image[0]
    h, w = image.shape[1:]
    new_h, new_w = output_size, output_size

    top = (h - new_h) // 2
    left = (w - new_w) // 2

    image = image[:, top:top + new_h, left:left + new_w]
    # print('output image shape:', image.shape)
    return image

def make_dir(dir_path):
    try:
        os.makedirs(dir_path)
    except OSError:
        pass
    return dir_path

def update_env_kwargs(vv):
    new_vv = vv.copy()
    for v in vv:
        if v.startswith('env_kwargs_'):
            arg_name = v[len('env_kwargs_'):]
            new_vv['env_kwargs'][arg_name] = vv[v]
            del new_vv[v]
    return new_vv

class NumpyEncoder(json.JSONEncoder):
    """ Encode numpy arrays as lists. Use with 'cls' kwarg:
    numpyData = {"array": my_numpy_array}
    encodedNumpyData = json.dumps(numpyData, cls=NumpyArrayEncoder) # dump to string
    json.dump(numpyData, file, cls=NumpyArrayEncoder) # dump to file
    """
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.int64):
            return int(obj)
        return json.JSONEncoder.default(self, obj)

def fill_replay_buffer(rsi_file, model):
    """
    Fill replay buffer with transitions from rsi_file (reference states).
    """
    reference_states = np.load(rsi_file, allow_pickle=True)
    states = reference_states['ob_trajs']
    next_states = reference_states['ob_next_trajs']
    actions = reference_states['action_trajs']
    rewards = reference_states['reward_trajs']
    dones = reference_states['done_trajs']
    for ep_counter in range(states.shape[0]):
        for traj_counter in range(len(states[ep_counter])):
            model.replay_buffer.add(
                states[ep_counter][traj_counter],
                next_states[ep_counter][traj_counter],
                actions[ep_counter][traj_counter],
                rewards[ep_counter][traj_counter],
                dones[ep_counter][traj_counter],
                [dict()],
            )