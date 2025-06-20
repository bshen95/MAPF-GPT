import requests
import re
import yaml
import os
import zipfile
import shutil
import sys

BASE_BEN_URL = "https://tracker.pathfinding.ai/quickDownload/benchmarks/"
BASE_RES_URL = "https://tracker.pathfinding.ai/quickDownload/results/"
BASE_DIR = "movingAI"
SCENARIO_DIR = BASE_DIR + "/scenarios"
RESULTS_DIR = BASE_DIR + "/results"
MAPS_YAML = BASE_DIR + "/maps_config.yaml"
from pogema_toolbox.generators.generator_utils import maps_dict_to_yaml


maps_config = {
    "maps": [
        # "Berlin_1_256",
        # "Boston_0_256",
        # "Paris_1_256",
        # "brc202d",
        # "den312d",
        # "den520d",
        # "empty-16-16",
        # "empty-32-32",
        # "empty-48-48",
        # "empty-8-8",
        # "ht_chantry",
        # "ht_mansion_n",
        # "lak303d",
        # "lt_gallowstemplar_n",
        # "maze-128-128-1",
        # "maze-128-128-10",
        # "maze-128-128-2",
        # "maze-32-32-2",
        # "maze-32-32-4",
        # "orz900d",
        # "ost003d",
        # "random-32-32-10",
        "random-32-32-20",
        "random-64-64-10",
        "random-64-64-20",
        "room-32-32-4",
        "room-64-64-16",
        "room-64-64-8",
        # "w_woundedcoast",
        # "warehouse-10-20-10-2-1",
        # "warehouse-10-20-10-2-2",
        # "warehouse-20-40-10-2-1",
        # "warehouse-20-40-10-2-2"
    ]
}









def download_moving_ai_maps(url):
    response = requests.get(url)

    zip_file = io.BytesIO(response.content)

    z = zipfile.ZipFile(zip_file, 'r')

    maps_dict = {}

    for file_name in z.namelist():
        if file_name.endswith('.map'):
            with z.open(file_name) as f:
                grid = map_to_grid(f)
                maps_dict[file_name.replace('.map', "")] = grid

    z.close()

    return maps_dict


def map_to_grid(file, remove_border=False):
    lines = []
    with file as f:
        type_ = f.readline().split(' ')[1]
        height = int(f.readline().split(' ')[1])
        width = int(f.readline().split(' ')[1])
        _ = f.readline()

        for _ in range(height):
            line = f.readline().rstrip()
            lines.append(line)

    m = []
    rmb = 1 if remove_border else 0
    for i in range(rmb, len(lines) - rmb):
        line = []
        for j in range(rmb, len(lines[i]) - rmb):
            symbol = lines[i][j]
            is_obstacle = symbol in ['@', 'O', 'T']
            line.append('#' if is_obstacle else '.')
        m.append("".join(line))
    return '\n'.join(m)


def generate_maps_yaml(output_path=MAPS_YAML):
    ensure_dirs()
    # 1. Download index page
    resp = requests.get(BASE_BEN_URL)
    resp.raise_for_status()

    # 2. Find all .zip links
    zips = re.findall(r'href=["\']([^"\']+\.zip)["\']', resp.text)
    if not zips:
        print("No .zip files found at", BASE_BEN_URL)
        return

    # 3. Extract map names by removing path and .zip
    map_names = [re.sub(r'^.*/|\.zip$', '', f) for f in zips]

    # Remove duplicates and sort
    map_names = sorted(set(map_names))

    # 4. Build YAML structure
    data = {"maps": map_names}

    # 5. Write to YAML file
    with open(output_path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)

    print(f"Wrote {len(map_names)} map(s) to {output_path}")


def download_file(url, dest_path):
    r = requests.get(url, stream=True)
    if r.status_code == 200:
        with open(dest_path, "wb") as f:
            for chunk in r.iter_content(8192):
                f.write(chunk)
        return True
    else:
        print(f"Failed to download {url} - Status code {r.status_code}")
        return False

def unzip_file(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

def ensure_dirs():
    os.makedirs(BASE_DIR, exist_ok=True)
    os.makedirs(SCENARIO_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

def download_scenarios_and_results(input_path=maps_config):
    ensure_dirs()

    # Load map names from yaml
    # with open(input_path) as f:
    #     data = yaml.safe_load(f)

    data = maps_config
    maps = data.get("maps", [])
    if not maps:
        print("No maps found in maps.yaml")
        return

    for map_name in maps:
        print(f"\nProcessing map: {map_name}")

        # Download zip to scenarios/
        zip_url = f"{BASE_BEN_URL}{map_name}.zip"
        zip_path = os.path.join(SCENARIO_DIR, f"{map_name}.zip")
        if download_file(zip_url, zip_path):
            print(f"Downloading scenario for {map_name}")
            # Unzip into scenarios/
            unzip_file(zip_path, SCENARIO_DIR)
            os.remove(zip_path)
            print(f"Extracted and removed zip for {map_name}")

        # Download CSV to results/
        results_url = f"{BASE_RES_URL}{map_name}.zip"
        results_path = os.path.join(RESULTS_DIR, f"{map_name}.zip")
        if download_file(results_url, results_path):
            # Unzip into scenarios/
            print(f"Downloading result for {map_name}")
            unzip_file(results_path, RESULTS_DIR)
            os.remove(results_path)
            print(f"Extracted and removed zip for {map_name}")

    maps_dict = {}
    for map_name in maps:
        print(f"\nRegistering map: {map_name}")
        map_file_name = f"{SCENARIO_DIR}/{map_name}/{map_name}.map"
        with open(map_file_name) as f:
                grid = map_to_grid(f)
                maps_dict[map_name] = grid
    maps_dict_to_yaml(BASE_DIR+'/maps.yaml', maps_dict)

    maps_config["maps"] = [m for m in maps_config["maps"] if not str(m).startswith("#")]

    # Save to maps.yaml
    with open(MAPS_YAML, "w") as f:
        yaml.safe_dump(maps_config, f, sort_keys=False)

    print(f"\nExported config file to {MAPS_YAML}")

def clean():
    # Remove results directory
    if os.path.isdir(BASE_DIR):
        shutil.rmtree(BASE_DIR)
        print(f"Removed directory {BASE_DIR}")
    else:
        print(f"{BASE_DIR} directory does not exist")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "clean":
        clean()
    elif len(sys.argv) > 1 and sys.argv[1] == "gen":
        download_scenarios_and_results()
    else:
        download_scenarios_and_results()