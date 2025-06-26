import glob
import os
import json
import yaml
import hashlib
import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc
import shutil
import random
import multiprocessing as mp
from functools import partial
from pathlib import Path

from pogema_toolbox.create_env import Environment
from pogema_toolbox.eval_utils import initialize_wandb, save_evaluation_results
from cus_evalution import evaluation
from pogema_toolbox.registry import ToolboxRegistry
from collections import defaultdict


from create_env import create_logging_env
from expert_inference import ExpertInference, ExpertInferenceConfig
from tokenizer.generate_observations import ObservationGenerator
from tokenizer.parameters import InputParameters
from download_results_scenario import download_scenarios_and_results
import zipfile
from pathlib import Path
import csv
import sys

csv.field_size_limit(sys.maxsize)

BASE_DIR = "movingAI"
SCENARIO_DIR = BASE_DIR + "/scenarios"
RESULTS_DIR = BASE_DIR + "/results"
MAPS_YAML = BASE_DIR + "/maps_config.yaml"
EXPERT_DATA_FOLDER = "expert_data"

DATASET_FOLDER = "dataset"
TEMP_FOLDER = "temp"
NUM_PROCESSES = 12
NUM_SPLITS = 24


MAP_CONFIG = {
    "maps": [
        {"name": "Berlin_1_256", "type": "city"},
        {"name": "Boston_0_256", "type": "city"},
        {"name": "Paris_1_256", "type": "city"},
        {"name": "ht_chantry", "type": "game"},
        {"name": "ht_mansion_n", "type": "game"},
        {"name": "lak303d", "type": "game"},
        {"name": "lt_gallowstemplar_n", "type": "game"},
        {"name": "brc202d", "type": "game"},
        {"name": "den312d", "type": "game"},
        {"name": "den520d", "type": "game"},
        {"name": "orz900d", "type": "game"},
        {"name": "ost003d", "type": "game"},
        {"name": "w_woundedcoast", "type": "game"},
        {"name": "empty-16-16", "type": "empty"},
        {"name": "empty-32-32", "type": "empty"},
        {"name": "empty-48-48", "type": "empty"},
        {"name": "empty-8-8", "type": "empty"},
        {"name": "maze-128-128-1", "type": "maze"},
        {"name": "maze-128-128-10", "type": "maze"},
        {"name": "maze-128-128-2", "type": "maze"},
        {"name": "maze-32-32-2", "type": "maze"},
        {"name": "maze-32-32-4", "type": "maze"},
        {"name": "random-32-32-10", "type": "random"},
        {"name": "random-32-32-20", "type": "random"},
        {"name": "random-64-64-10", "type": "random"},
        {"name": "random-64-64-20", "type": "random"},
        {"name": "room-32-32-4", "type": "room"},
        {"name": "room-64-64-16", "type": "room"},
        {"name": "room-64-64-8", "type": "room"},
        {"name": "warehouse-10-20-10-2-1", "type": "warehouse"},
        {"name": "warehouse-10-20-10-2-2", "type": "warehouse"},
        {"name": "warehouse-20-40-10-2-1", "type": "warehouse"},
        {"name": "warehouse-20-40-10-2-2", "type": "warehouse"}
    ]
}

RANDOM_MAPS_FOLDER = "dataset_configs/12-medium-random"
MAZES_MAPS_FOLDER = "dataset_configs/10-medium-mazes"

NUM_CHUNKS = 50
FILE_PER_CHUNK = 10
DESIRED_SIZE = 10*2**21 # per chunk
MAZE_RATIO = 0.9
NUM_PROCESSES = 12

def tensor_to_hash(tensor):
    tensor_bytes = tensor.tobytes()
    return hashlib.sha256(tensor_bytes).hexdigest()

def get_files_by_type(folder_path):
    all_files = glob.glob(os.path.join(folder_path, '*.json'))
    unique_types = set()

    for entry in MAP_CONFIG['maps']:
        map_type = entry['type']
        unique_types.add(map_type)

    # If you want a sorted list of unique types
    unique_types = sorted(list(unique_types))

    type_files = {}
    for map_type in unique_types:
        files = [f for f in all_files if map_type in os.path.basename(f).lower()]
        files.sort()
        if files:
            type_files[map_type] = files
    return type_files 

def generate_part(map_name, maps):
    print("processing map", map_name)
    cfg = InputParameters()
    with open(map_name, "r") as f:
        data = json.load(f)
    generator = ObservationGenerator(maps, data, cfg)
    tensors, gt_actions = generator.generate_observations(0, len(data))
    return tensors, gt_actions

def balance_and_filter_tensors(tensors, actions, known_hashes=None):
    new_tensors = []
    new_actions = []
    duplicates = 0
    if known_hashes is None:
        known_hashes = set()
    for tensor, action in zip(tensors, actions):
        tensor_hash = tensor_to_hash(tensor)

        if tensor_hash not in known_hashes:
            known_hashes.add(tensor_hash)
            new_tensors.append(tensor)
            new_actions.append(action)
        else:
            duplicates += 1
    if len(new_tensors) > 0:
        actions_made = [0 for i in range(6)]
        for action in new_actions:
            actions_made[action] += 1
        i = len(new_tensors) - 1
        discarded = 0
        while i >= 0:
            if new_actions[i] == 5:
                if (actions_made[0] + actions_made[5]) > len(new_tensors) // 5:
                    new_actions.pop(i)
                    new_tensors.pop(i)
                    actions_made[5] -= 1
                    discarded += 1
                else:
                    new_actions[i] = 0
            i -= 1
        print(discarded, duplicates, len(new_tensors), actions_made)
        new_tensors = np.array(new_tensors)
        new_actions = np.array(new_actions)
        indices = np.arange(len(new_tensors))
        np.random.shuffle(indices) # shuffle to balance actions
        new_tensors = new_tensors[indices]
        new_actions = new_actions[indices]
    return new_tensors, new_actions

def calculate_elements_to_pick(data, total_pick_count):
    file_elements = {}
    total_elements = 0
    for file, (tensors, actions) in data.items():
        total_elements += len(tensors)
        file_elements[file] = len(tensors)
    if total_pick_count > total_elements:
        print(
            f"Warning! Files don't contain enough data to pick {total_pick_count} elements. Using {total_elements} elements instead"
        )
        total_pick_count = total_elements

    elements_to_pick = {}
    total_picked = 0
    for file_path, num_elements in file_elements.items():
        elements_to_pick[file_path] = int(
            num_elements * total_pick_count / total_elements
        )
        total_picked += elements_to_pick[file_path]

    while total_picked < total_pick_count:
        for file_path, num_elements in file_elements.items():
            if total_picked == total_pick_count:
                break
            if elements_to_pick[file_path] < num_elements:
                elements_to_pick[file_path] += 1
                total_picked += 1

    return elements_to_pick, total_pick_count



def generate_chunks():
    type_files = get_files_by_type(TEMP_FOLDER)
    file_chunks = []
    for map_type, files in type_files.items():
        file_chunk_size = len(files) // NUM_CHUNKS
        chunks = [files[i:i+file_chunk_size] for i in range(0, len(files), file_chunk_size)]
        file_chunks.append({map_type: chunks})


    # for i in range(NUM_CHUNKS):
    #     process_files(maze_chunks[i], random_chunks[i], f"{DATASET_FOLDER}/chunk_{i}")


    # maze_files, random_files = get_files_by_type(TEMP_FOLDER)
    
    # maze_chunk_size = len(maze_files) // NUM_CHUNKS
    # random_chunk_size = len(random_files) // NUM_CHUNKS
    
    # maze_chunks = [maze_files[i:i+maze_chunk_size] for i in range(0, len(maze_files), maze_chunk_size)]
    # random_chunks = [random_files[i:i+random_chunk_size] for i in range(0, len(random_files), random_chunk_size)]
    
    # for i in range(NUM_CHUNKS):
    #     process_files(maze_chunks[i], random_chunks[i], f"{DATASET_FOLDER}/chunk_{i}")
        

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
    results.sort(key=lambda x: x["scen_path"])
    return results

def load_solution_csv_grouped(csv_path):
    grouped_results = defaultdict(list)

    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            map_name = row["map_name"]
            scen_type = row["scen_type"]
            type_id = row["type_id"]

            scen_path = f"{map_name}-{scen_type}-{type_id}.scen"
            agents = int(row["agents"])
            plan_lines = row["solution_plan"]

            grouped_results[scen_path].append({
                "agents": agents,
                "solution_plan": plan_lines
            })
    
    return dict(grouped_results)



def tracker_data_2_pogema():
    ToolboxRegistry.register_env("Environment", create_logging_env, Environment)
    ToolboxRegistry.register_algorithm("Expert", ExpertInference, ExpertInferenceConfig)
    maps = load_yaml_file("movingAI/maps.yaml")
    ToolboxRegistry.register_maps(maps)
    map_configs = load_yaml_file("movingAI/maps_config.yaml")

    for map_name in map_configs['maps']:
        print(f"Processing the map: {map_name}")
        # process_per_scenarios_file(map_name)
        process_per_map(map_name)



def load_yaml_file(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)



def process_per_map(map_name):
    scen_dir = os.path.join(SCENARIO_DIR, map_name)
    scen_files = [f for f in os.listdir(scen_dir) if f.endswith(".scen")]

    if not scen_files:
        print(f"No .scen files found in {scen_dir}")
        return

    st_dict = {
        scen_file: parse_scen_file(os.path.join(scen_dir, scen_file))
        for scen_file in scen_files
    }

    result_path = f"{RESULTS_DIR}/{map_name}.csv"
    results = load_solution_csv(result_path)
    process_map_results(map_name, st_dict, results)
    



def process_map_results(map_name, map_tasks, results):
    env_configs = []
    env_grid_searches = []
    algo_configs = []

    for result in results:
        scen_path = result['scen_path']
        agents = result["agents"]
        solution_plan = result["solution_plan"]

        selected = map_tasks[scen_path][:agents]
        starts = [item['start'] for item in selected]
        targets = [item['goal'] for item in selected]

        env_configs.append(create_env_config(map_name, agents, starts, targets))
        env_grid_searches.append(create_env_grid_search(map_name, agents))
        algo_configs.append(create_algo_config(solution_plan))

    key = 'Expert'
    eval_dir = Path(EXPERT_DATA_FOLDER) / map_name 
    initialize_wandb(algo_configs[-1], eval_dir, False, EXPERT_DATA_FOLDER)
    evaluation(env_configs, env_grid_searches, algo_configs, key, eval_dir)
    save_evaluation_results(eval_dir)


def process_per_scenarios_file(map_name):
    scen_dir = os.path.join(SCENARIO_DIR, map_name)
    scen_files = [f for f in os.listdir(scen_dir) if f.endswith(".scen")]

    if not scen_files:
        print(f"No .scen files found in {scen_dir}")
        return

    st_dict = {
        scen_file: parse_scen_file(os.path.join(scen_dir, scen_file))
        for scen_file in scen_files
    }

    result_path = f"{RESULTS_DIR}/{map_name}.csv"
    loaded_results = load_solution_csv_grouped(result_path)

    for scen_path, results in loaded_results.items():
        if scen_path not in st_dict:
            print(f"Warning: {scen_path} not found in scenario tasks.")
            continue

        process_scenario_results(map_name, scen_path, st_dict[scen_path], results)


def process_scenario_results(map_name, scen_path, scenario_tasks, results):
    env_configs = []
    env_grid_searches = []
    algo_configs = []

    for result in results:
        agents = result["agents"]
        solution_plan = result["solution_plan"]

        selected = scenario_tasks[:agents]
        starts = [item['start'] for item in selected]
        targets = [item['goal'] for item in selected]

        env_configs.append(create_env_config(map_name, agents, starts, targets))
        env_grid_searches.append(create_env_grid_search(map_name, agents))
        algo_configs.append(create_algo_config(solution_plan))

    key = 'Expert'
    eval_dir = Path(EXPERT_DATA_FOLDER) / map_name / scen_path.replace(".scen", "")
    initialize_wandb(algo_configs[-1], eval_dir, False, EXPERT_DATA_FOLDER)
    evaluation(env_configs, env_grid_searches, algo_configs, key, eval_dir)
    save_evaluation_results(eval_dir)


def create_env_config(map_name, agents, starts, targets):
    return {
        'name': 'Environment',
        'with_animation': False,
        'on_target': 'nothing',
        'max_episode_steps': 512,
        'observation_type': 'MAPF',
        'collision_system': 'soft',
        'seed': 0,
        'size': 1024,
        'num_agents': agents,
        'agents_xy': starts,
        'targets_xy': targets,
        'map_name': map_name
    }


def create_algo_config(solution_plan):
    return {
        'name': 'Expert',
        'time_limit': 10,
        'timeouts': [10],
        'num_process': 12,
        'parallel_backend': 'balanced_dask',
        'path': solution_plan
    }


def create_env_grid_search(map_name, agents):
    return {
        ('seed',): 0,
        ('num_agents',): agents,
        ('map_name',): map_name
    }




def process_file(file, maps):
    tensors, actions = generate_part(file, maps)
    tensors, actions = balance_and_filter_tensors(tensors, actions)
    return file, tensors, actions


def process_entire_files(map_data, map_file, output_file):
    # process the entire file, and do not remove data from file.
    # Process map files
    file, all_tensors, all_actions = process_file(map_file, map_data)
    # Combine results into dictionaries

    # Shuffle the data
    indices = np.arange(len(all_tensors))
    np.random.shuffle(indices)
    # Save the data
    all_tensors = all_tensors[indices]
    all_actions = all_actions[indices]

    # Define schema
    schema = pa.schema([
        ('input_tensors', pa.list_(pa.int8())),
        ('gt_actions', pa.int8())
    ])

    # Convert to Arrow arrays
    input_tensors_col = pa.array(all_tensors.tolist(), type=pa.list_(pa.int8()))
    gt_actions_col = pa.array(all_actions)

    # Create Arrow table
    table = pa.Table.from_arrays([input_tensors_col, gt_actions_col], schema=schema)

    # Save to one .arrow file
    chunk_output_file = f"{output_file}.arrow"
    with open(chunk_output_file, "wb") as f:
        with ipc.new_file(f, schema) as writer:
            writer.write(table)

    print(f"Saved {chunk_output_file} with {len(all_tensors)} samples")


def split_json(args):
    """Worker function to write a JSON chunk to a file."""
    chunk, out_path = args
    with open(out_path, 'w', encoding='utf-8') as out_f:
        json.dump(chunk, out_f, ensure_ascii=False, indent=2)
    return str(out_path)

def split_zipped_json_array(zip_path, output_dir, n):
    """
    Splits a zipped JSON array file into `n` subfiles using multiprocessing.

    Args:
        zip_path (str or Path): Path to the input .zip file containing a single JSON file.
        output_dir (str or Path): Directory where output subfiles will be stored.
        n (int): Number of chunks to split into.

    Returns:
        List of output file paths.
    """
    zip_path = Path(zip_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    with zipfile.ZipFile(zip_path, 'r') as zf:
        json_files = [name for name in zf.namelist() if name.endswith('.json')]
        if not json_files:
            raise ValueError(f"No .json file found inside zip: {zip_path}")
        json_filename = json_files[0]

        with zf.open(json_filename) as f:
            raw = f.read()
            if not raw.strip():
                raise ValueError(f"The file {json_filename} inside the zip is empty.")
            f.seek(0)
            try:
                data = json.loads(raw)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in {json_filename}: {e}") from e

    if not isinstance(data, list):
        raise TypeError(f"Expected a JSON array, got: {type(data).__name__}")

    # Shuffle and prepare chunks
    total = len(data)
    random.shuffle(data)
    chunk_size = (total + n - 1) // n

    jobs = []
    for i in range(n):
        chunk = data[i * chunk_size : (i + 1) * chunk_size]
        if not chunk:
            continue
        out_path = output_dir / f"{zip_path.stem}_part_{i}.json"
        jobs.append((chunk, out_path))

    # Ensure we use 12 workers
    with mp.Pool(processes=NUM_PROCESSES) as pool:
        output_files = pool.map(split_json, jobs)

    return output_files


def unzip_json(zip_path: str, output_path: str):
    """
    Extracts the first .json file from the zip archive at zip_path
    and saves it to output_path.
    Creates output directory if it doesn't exist.
    """
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, 'r') as z:
        for file_name in z.namelist():
            if file_name.endswith('.json'):
                with z.open(file_name) as src, open(output_path, 'wb') as dst:
                    dst.write(src.read())
                return  # stop after extracting the first json file
    raise FileNotFoundError(f"No .json file found in {zip_path}")

def process_individual_map(map_name):
    maps= yaml.safe_load(open(f"movingAI/maps.yaml", "r"))
    map_chunks = []
    for i in range(1,25):
        # print(i)
        unzip_json(f"expert_data/{map_name}/{map_name}-even-{i}.zip", f"{TEMP_FOLDER}/{map_name}-even-{i}.json")
        map_chunks.append(f"{TEMP_FOLDER}/{map_name}-even-{i}.json")
        unzip_json(f"expert_data/{map_name}/{map_name}-random-{i}.zip", f"{TEMP_FOLDER}/{map_name}-random-{i}.json")
        map_chunks.append(f"{TEMP_FOLDER}/{map_name}-random-{i}.json")

    Path(DATASET_FOLDER).mkdir(parents=True, exist_ok=True)
    for i in range(NUM_CHUNKS):
        print(f"{DATASET_FOLDER}/{map_chunks[i][5:-5]}")
        process_entire_files( maps, map_chunks[i], f"{DATASET_FOLDER}/{map_chunks[i][5:-5]}")

    shutil.rmtree(TEMP_FOLDER)



    # print(f"maps_random: {maps_random}")
    # print(f"random_files: {random_files}")
    # with mp.Pool(NUM_PROCESSES) as pool:
    #     random_results = pool.map(partial(process_file, maps=maps_random),random_files)
    # stop = 0 




    # maps_mazes = yaml.safe_load(open(f"{MAZES_MAPS_FOLDER}/maps.yaml", "r"))
    # maze_desired_size = int(DESIRED_SIZE * MAZE_RATIO)
    # random_desired_size = DESIRED_SIZE - maze_desired_size
    
    # # Process maze files
    # with mp.Pool(NUM_PROCESSES) as pool:
    #     maze_results = pool.map(partial(process_file, maps=maps_mazes), maze_files)
    
    # # Process random files
    # with mp.Pool(NUM_PROCESSES) as pool:
    #     random_results = pool.map(partial(process_file, maps=maps_random), random_files)
    
    # # Combine results into dictionaries
    # maze_data = {file: (tensors, actions) for file, tensors, actions in maze_results}
    # random_data = {file: (tensors, actions) for file, tensors, actions in random_results}

    
    # # Pick required portion from each file
    # maze_elements_to_pick, total_maze_elements = calculate_elements_to_pick(maze_data, maze_desired_size)
    # random_elements_to_pick, total_random_elements = calculate_elements_to_pick(random_data, random_desired_size)
    
    # all_tensors = np.empty((total_maze_elements + total_random_elements, 256), dtype=np.int8)
    # all_actions = np.empty(total_maze_elements + total_random_elements, dtype=np.int8)
    
def worker(args):
    maps, temp_path, output_path = args
    process_entire_files(maps, temp_path, output_path)


def pogema_2_OA_pairs():
    maps = load_yaml_file("movingAI/maps.yaml")
    map_configs = load_yaml_file("movingAI/maps_config.yaml")
    Path(DATASET_FOLDER).mkdir(parents=True, exist_ok=True) 
    Path(TEMP_FOLDER).mkdir(parents=True, exist_ok=True)
    for map_name in map_configs['maps']:
        zip_path = Path(EXPERT_DATA_FOLDER) / f"{map_name}.zip"
        print(zip_path)
        print(f"Processing the map: {map_name}")
        NUM_SPLITS = 2 * NUM_PROCESSES
        split_zipped_json_array(zip_path, TEMP_FOLDER, NUM_SPLITS)
        args_list = [
            (maps,
            f"{TEMP_FOLDER}/{map_name}_part_{i}.json",
            f"{DATASET_FOLDER}/{map_name}_part_{i}")
            for i in range(NUM_SPLITS)
        ]
        with mp.Pool(processes=NUM_PROCESSES) as pool:
            pool.map(worker, args_list)
        # process_per_scenarios_file(map_name)
        # process_per_map(map_name)


def main():
    # Step 1: Download scenarios and results from tracker.
    # maps_config = {"maps": [entry["name"] for entry in MAP_CONFIG["maps"]] }

    # download_scenarios_and_results(maps_config)
    # load_movingAI_maps()
    # tracker_data_2_pogema()
    pogema_2_OA_pairs()
    # process_individual_map("room-32-32-4")
    # # Step 2: Convert the csv to MAPF-GPT format.
    # # load_movingAI_maps()
    # files = [f"{EXPERT_DATA_FOLDER}/{m['name']}/Expert.json" for m in MAP_CONFIG['maps']]

    # # this function split the map-based json 
    # with mp.Pool() as pool:

    # pool.map(split_json, files)




if __name__ == "__main__":
    main()
