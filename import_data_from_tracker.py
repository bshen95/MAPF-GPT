import glob
import os
import json
import yaml
import hashlib
import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc
import shutil
import multiprocessing as mp
from functools import partial
from pathlib import Path
from typing import Literal
from pogema_toolbox.create_env import Environment
from pogema_toolbox.eval_utils import initialize_wandb, save_evaluation_results
from pogema_toolbox.evaluator import evaluation
from pogema_toolbox.registry import ToolboxRegistry
from pogema_toolbox.evaluator import balanced_dask_backend
from pogema_toolbox.results_holder import ResultsHolder
from pogema_toolbox.algorithm_config import AlgoBase
from pogema import GridConfig
from pydantic import Extra
from create_env import create_logging_env
from tokenizer.generate_observations import ObservationGenerator
from tokenizer.parameters import InputParameters

DATASET_FOLDER = "dataset"
CONFIGS = [
    "dataset_configs/10-medium-mazes/10-medium-mazes-part1.yaml",
    # "dataset_configs/10-medium-mazes/10-medium-mazes-part2.yaml",
    # "dataset_configs/10-medium-mazes/10-medium-mazes-part3.yaml",
    # "dataset_configs/10-medium-mazes/10-medium-mazes-part4.yaml",
    # "dataset_configs/12-medium-random/12-medium-random-part1.yaml",
]

def run_episode(env, algo):
    """
    Runs an episode in the environment using the given algorithm.

    Args:
        env: The environment to run the episode in.
        algo: The algorithm used for action selection.

    Returns:
        ResultsHolder: Object containing the results of the episode.
    """
    algo.reset_states()
    results_holder = ResultsHolder()

    obs, _ = env.reset(seed=env.grid_config.seed)
    while True:
        actions = algo.act(obs)
        obs, rew, terminated, truncated, infos = env.step(actions)
        results_holder.after_step(infos)

        if all(terminated) or all(truncated):
            break
    return results_holder.get_final()






# used for simulating the expert data. 
class ExpertAgent:
    def __init__(self, idx):
        self._moves = GridConfig().MOVES
        self._reverse_actions = {tuple(self._moves[i]): i for i in range(len(self._moves))}

        self.idx = idx
        self.previous_goal = None
        self.path = []

    def is_new_goal(self, new_goal):
        return not self.previous_goal == new_goal
    
    def set_new_goal(self, new_goal):
        self.previous_goal = new_goal

    def set_path(self, new_path):
        self.path = new_path[::-1]

    def get_action(self):
        action = 0
        if len(self.path) > 1:
           x, y = self.path[-1]
           tx, ty = self.path[-2]
           action = self._reverse_actions[tx - x, ty - y]
           self.path.pop()
        # print(action)
        return action

    def clear_state(self):
        self.previous_goal = None
        self.path = []

class ExpertInferenceConfig(AlgoBase, extra=Extra.forbid):
    name: Literal['Expert'] = 'Expert'
    time_limit: float = 60
    timeouts: list = [1.0, 5.0, 10.0, 60.0]
    path: str = ""

class ExpertInference:
    def __init__(self, cfg: ExpertInferenceConfig):
        self.cfg = cfg
        self.expert_agents = None
        self.path = cfg.path
        
    def _convert_path(self, agent_starts_xy, agent_targets_xy):
        """
        Converts motion strings (self.path) to actual coordinate paths
        for each agent based on their start positions.
        
        Args:
            agent_starts_xy (List[Tuple[int, int]]): Starting coordinates of agents.
            agent_targets_xy (List[Tuple[int, int]]): Target coordinates (not used here but may be useful for validation).

        Returns:
            List[List[Tuple[int, int]]]: List of paths (each a list of coordinates) for each agent.
        """
        # Define movement deltas
        direction_delta = {
            'u': (0, 1),
            'd': (0, -1),
            'l': (-1, 0),
            'r': (1, 0),
            'w': (0, 0)
        }

        agent_motion_strings = self.path.strip().split("\n")
        all_paths = []

        for i, motion_str in enumerate(agent_motion_strings):
            x, y = agent_starts_xy[i]
            path = [(x, y)]

            for move in motion_str:
                dx, dy = direction_delta[move]
                x += dx
                y += dy
                path.append((x, y))

            all_paths.append(path)

        return all_paths
    

    def act(self, observations, rewards=None, dones=None, info=None, skip_agents=None):
        map_array = np.array(observations[0]['global_obstacles'])
        agent_starts_xy = [obs['global_xy'] for obs in observations]
        agent_targets_xy = [obs['global_target_xy'] for obs in observations]
        if self.expert_agents is None:
            self.expert_agents = [ExpertAgent(idx) for idx in range(len(observations))]
            for idx, (start_xy, target_xy) in enumerate(zip(agent_starts_xy, agent_targets_xy)):
                self.expert_agents[idx].set_new_goal(target_xy)
            converted_paths = self._convert_path(agent_starts_xy, agent_targets_xy)
            for idx, agent_path in enumerate(converted_paths):
                self.expert_agents[idx].set_path(agent_path)

        return [agent.get_action() for agent in self.expert_agents]

    def after_step(self, dones):
        pass

    def reset_states(self):
        self.expert_agents = None

    def after_reset(self):
        pass

    def get_additional_info(self):
        addinfo = {"rl_used": 0.0}
        return addinfo

def simluate_expert_algorithm():
    env_cfg_name = "Environment"
    ToolboxRegistry.register_env(env_cfg_name, create_logging_env, Environment)
    ToolboxRegistry.register_algorithm("Expert", ExpertInference, ExpertInferenceConfig)
    unique_paths = {os.path.dirname(path) for path in CONFIGS}
    maps = {}
    for path in unique_paths:
        with open(f"{path}/maps.yaml", "r") as f:
            folder_maps = yaml.safe_load(f)
            maps.update(folder_maps)
    ToolboxRegistry.register_maps(maps)
    
    env_config = ({'name': 'Environment', 'with_animation': False, 'on_target': 'nothing', 'max_episode_steps': 128, 
                   'observation_type': 'MAPF', 'collision_system': 'soft', 
                   'seed': 0, 'num_agents': 2, 'agents_xy': [[4, 4], [2, 2]],
                   'targets_xy': [[0, 0], [1, 0]], 
                   'map_name': 'medium-mazes-seed-0000'})
    algo_config = ( {'name': 'Expert', 'time_limit': 10, 'timeouts': [10], 'num_process': 60, 'parallel_backend': 'balanced_dask',
                     'path': 'ldurw\nldurw\n'
                     })
    #observation_size = 5 that's why it is start from 5.
    algo = ToolboxRegistry.create_algorithm("Expert", **algo_config)
    env = ToolboxRegistry.create_env(env_config['name'], **env_config)
    results = []
    results.append(run_episode(env, algo))
    print(results)


def main():
    # Step 1: Run LaCAM to obtain expert data in json format.
    simluate_expert_algorithm()



if __name__ == "__main__":
    main()
