import os.path as osp
import argparse
import numpy as np

from softgym.registered_env import env_arg_dict, SOFTGYM_ENVS
from softgym.envs.rope_flatten import RopeFlattenEnv
from softgym.utils.normalized_env import normalize
from softgym.utils.visualization import save_numpy_as_gif

from sb3 import utils

def main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    # ['PassWater', 'PourWater', 'PourWaterAmount', 'RopeFlatten', 'ClothFold', 'ClothFlatten', 'ClothDrop', 'ClothFoldCrumpled', 'ClothFoldDrop', 'RopeConfiguration']
    parser.add_argument('--env_name', type=str, default='RopeFlatten')
    parser.add_argument('--output_folder', type=str, default='./data/SOTA_Table_Evaluations/RopeFlatten/expert_policy')
    parser.add_argument('--headless', type=int, default=1, help='Whether to run the environment with headless rendering')
    parser.add_argument('--num_variations', type=int, default=1000, help='Number of environment variations to be generated')

    args = parser.parse_args()

    env_kwargs = env_arg_dict[args.env_name]

    env_kwargs['use_cached_states'] = True
    env_kwargs['save_cached_states'] = False
    env_kwargs['num_variations'] = args.num_variations
    env_kwargs['render'] = True
    env_kwargs['headless'] = args.headless

    env = normalize(SOFTGYM_ENVS[args.env_name](**env_kwargs))

    utils.make_dir(args.output_folder)

    total_normalized_perf_final = []
    random_seeds = [100, 201, 302, 403, 504]
    for curr_seed in random_seeds:
        utils.set_seed_everywhere(curr_seed)
        for episode in range(20):
            env.reset()
            done = False
            ep_normalized_perf = []
            while not done:
                action = env.compute_expert_action()
                _, _, done, info = env.step(action)
                ep_normalized_perf.append(info['normalized_performance'])
            total_normalized_perf_final.append(ep_normalized_perf[-1])
            print(f'Ep {episode}  Episode normalized performance final: {ep_normalized_perf[-1]}')

    total_normalized_perf_final = np.array(total_normalized_perf_final)

    np.save(args.output_folder + '/expert_total_normalized_perf_final.npy', total_normalized_perf_final)
    print('!!!!!!! info_normalized_performance_final !!!!!!!')
    print(f'Mean: {np.mean(total_normalized_perf_final):.4f}')
    print(f'Std: {np.std(total_normalized_perf_final):.4f}')
    print(f'Median: {np.median(total_normalized_perf_final):.4f}')
    print(f'25th Percentile: {np.percentile(total_normalized_perf_final, 25):.4f}')
    print(f'75th Percentile: {np.percentile(total_normalized_perf_final, 75):.4f}')

if __name__ == '__main__':
    main()
