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

from pogema_toolbox.create_env import Environment
from pogema_toolbox.eval_utils import initialize_wandb, save_evaluation_results
from cus_evalution import evaluation
from pogema_toolbox.registry import ToolboxRegistry



from create_env import create_logging_env
from expert_inference import ExpertInference, ExpertInferenceConfig
from tokenizer.generate_observations import ObservationGenerator
from tokenizer.parameters import InputParameters

import csv
import sys

csv.field_size_limit(sys.maxsize)

BASE_DIR = "movingAI"
SCENARIO_DIR = BASE_DIR + "/scenarios"
RESULTS_DIR = BASE_DIR + "/results"
MAPS_YAML = BASE_DIR + "/maps_config.yaml"
EXPERT_DATA_FOLDER = "Expert"

DATASET_FOLDER = "dataset"
CONFIGS = [
    "dataset_configs/10-medium-mazes/10-medium-mazes-part1.yaml"
]



# def simluate_expert_algorithm():
#     env_cfg_name = "Environment"
#     ToolboxRegistry.register_env(env_cfg_name, create_logging_env, Environment)
#     ToolboxRegistry.register_algorithm("Expert", ExpertInference, ExpertInferenceConfig)
#     unique_paths = {os.path.dirname(path) for path in CONFIGS}
#     maps = {}
#     for path in unique_paths:
#         with open(f"{path}/maps.yaml", "r") as f:
#             folder_maps = yaml.safe_load(f)
#             maps.update(folder_maps)
#     ToolboxRegistry.register_maps(maps)
    
#     env_config = ({'name': 'Environment', 'with_animation': False, 'on_target': 'nothing', 'max_episode_steps': 128, 
#                    'observation_type': 'MAPF', 'collision_system': 'soft', 
#                    'seed': 0, 'num_agents': 2, 'agents_xy': [[4, 4], [2, 2]],
#                    'targets_xy': [[0, 0], [1, 0]], 
#                    'map_name': 'medium-mazes-seed-0000'})
#     algo_config = ( {'name': 'Expert', 'time_limit': 10, 'timeouts': [10], 'num_process': 60, 'parallel_backend': 'balanced_dask',
#                      'path': 'ldurw\nldurw\n'
#                      })
#     #observation_size = 5 that's why it is start from 5.
#     algo = ToolboxRegistry.create_algorithm("Expert", **algo_config)
#     env = ToolboxRegistry.create_env(env_config['name'], **env_config)
#     results = []
#     results.append(run_episode(env, algo))
#     print(results)


def parse_scen_file(scen_path):
    tasks = []
    with open(scen_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        if line.startswith("version"):
            continue  # skip header if present

        parts = line.strip().split("\t")
        if len(parts) < 9:
            continue  # skip malformed lines

        start_x = int(parts[4])
        start_y = int(parts[5])
        goal_x = int(parts[6])
        goal_y = int(parts[7])

        tasks.append({
            "start": [start_y, start_x],
            "goal": [goal_y, goal_x]
        })

    return tasks

def load_solution_csv(csv_path):
    results = []
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            map_name = row["map_name"]
            scen_type = row["scen_type"]
            type_id = row["type_id"]

            scen_path = f"{map_name}-{scen_type}-{type_id}.scen"
            agents = int(row["agents"])
            plan_lines = row["solution_plan"]

            results.append({
                "scen_path": scen_path,
                "agents": agents,
                "solution_plan": plan_lines
            })
    return results






def load_movingAI_maps():
    env_cfg_name = "Environment"
    ToolboxRegistry.register_env(env_cfg_name, create_logging_env, Environment)
    ToolboxRegistry.register_algorithm("Expert", ExpertInference, ExpertInferenceConfig)
    maps = {}
    with open(f"movingAI/maps.yaml", "r") as f:
        folder_maps = yaml.safe_load(f)
        maps.update(folder_maps)
    ToolboxRegistry.register_maps(maps)
    map_configs = {}
    with open(f"movingAI/maps_config.yaml", "r") as f:
        folder_maps = yaml.safe_load(f)
        map_configs.update(folder_maps)

    env_configs = []
    env_grid_searches = []
    algo_configs = []
    for map_name in map_configs['maps']:
        scen_dir = os.path.join(SCENARIO_DIR, map_name)
        # Find all .scen files in the directory
        scen_files = [f for f in os.listdir(scen_dir) if f.endswith(".scen")]

        if not scen_files:
            print(f"No .scen files found in {scen_dir}")
            continue

        st_dict = {}
        for scen_file in scen_files:
            scen_path = os.path.join(scen_dir, scen_file)
            st_dict[scen_file] = parse_scen_file(scen_path)

        result_path = f"{RESULTS_DIR}/{map_name}.csv"
        loaded_results = load_solution_csv(result_path)
        for result in loaded_results:
            scen_path = result["scen_path"]
            agents = result["agents"]
            solution_plan = result["solution_plan"]

            if scen_path not in st_dict:
                print(f"Warning: {scen_path} not found in scenario tasks.")
                continue

            st = st_dict[scen_path]
            selected = st[:agents]
            starts = [item['start'] for item in selected]
            targets = [item['goal'] for item in selected]

            env_config = {
                'name': 'Environment',
                'with_animation': False,
                'on_target': 'nothing',
                'max_episode_steps': 128,
                'observation_type': 'MAPF',
                'collision_system': 'soft',
                'seed': 0,
                'size': 1024,
                'num_agents': agents,
                'agents_xy': starts,
                'targets_xy': targets,
                'map_name': map_name
            }
            algo_config = {
            'name': 'Expert',
            'time_limit': 10,
            'timeouts': [10],
            'num_process': 12,
            'parallel_backend': 'balanced_dask',
            'path': solution_plan
            }
            env_grid_search = {
                ('seed',): 0,
                ('num_agents',): agents,
                ('map_name',): map_name
            }
            env_configs.append(env_config)
            env_grid_searches.append(env_grid_search)
            algo_configs.append(algo_config)
        key = 'Expert'
        eval_dir = Path(EXPERT_DATA_FOLDER) / map_name
        initialize_wandb(algo_config, eval_dir, False, EXPERT_DATA_FOLDER)
        evaluation(env_configs, env_grid_searches, algo_configs, key, eval_dir)
        save_evaluation_results(eval_dir)















def main():
    # Step 1: Run LaCAM to obtain expert data in json format.
    # simluate_expert_algorithm()
    load_movingAI_maps()



if __name__ == "__main__":
    main()
