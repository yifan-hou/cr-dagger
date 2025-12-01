import sys
import os

SCRIPT_PATH = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(SCRIPT_PATH, "../../"))

import pathlib
import numpy as np
import zarr
import cv2
import concurrent.futures

CORRECTION = True   # set to true if you want to use the correction data

# check environment variables
if "PYRITE_RAW_DATASET_FOLDERS" not in os.environ:
    raise ValueError("Please set the environment variable PYRITE_RAW_DATASET_FOLDERS")
if "PYRITE_DATASET_FOLDERS" not in os.environ:
    raise ValueError("Please set the environment variable PYRITE_DATASET_FOLDERS")


# specify the input and output directories
id_list = [0]  # single robot
# id_list = [0, 1] # bimanual

ft_sensor_configuration = "handle_on_robot" # "handle_on_sensor" or "handle_on_robot"

output_dir = pathlib.Path(
os.environ.get("PYRITE_DATASET_FOLDERS") + "/online_belt_v2/raw"
)

# open the zarr store
store = zarr.DirectoryStore(path=output_dir)
root = zarr.open(store=store, mode="a")

print("Reading data from input_dir: ", output_dir)
episode_names = os.listdir(output_dir)

episode_config = {
    "output_dir": output_dir,
    "id_list": id_list,
    "ft_sensor_configuration": ft_sensor_configuration,
    "num_threads": 1,
    "has_correction": CORRECTION,
    "save_video": False,
    "max_workers": 32
}


import pandas as pd

def check_keys(episode_name, output_root, config):
    if episode_name.startswith("."):
        return True

    # print(f"[process_one_episode_into_zarr] episode_name: {episode_name}, episode_id: {episode_id}")
    episode_dir = pathlib.Path(config["output_dir"]).joinpath(episode_name)
    try:
        json_path = episode_dir.joinpath("key_data.json")
        df_key_data = pd.read_json(json_path)
        data_key = np.vstack(df_key_data["key_event"]).flatten()
        # Check if the data_key is empty
        if data_key.size == 0:
            print(f"\033[31mError: No key data found for {episode_dir}\033[0m.")
            return False
        if len(data_key) == 1 and data_key[0] == -1:
            print(f"\033[31mError: there's no correction for {episode_dir}\033[0m.")
            print(data_key)
            return False

        for i in range(0, len(data_key)-1, 2):
            if (data_key[i] != 1 and data_key[i] != -1) or (data_key[i + 1] != 0 and data_key[i + 1] != -1):
                print(f"\033[31mError: Key data is not in the correct format for {episode_dir}\033[0m.")
                print(data_key)
                return False
     
    except FileNotFoundError:
        # Handle the case where the file does not exist
        print(f"\033[31mError: JSON file not found for {episode_dir}\033[0m.")
    except ValueError as e:
        # Handle the case where the JSON is invalid
        print(f"\033[31mError: Invalid JSON format - {e} - for {episode_dir}\033[0m.")
    except Exception as e:
        # Handle other exceptions that might occur
        print(f"\033[31mAn unexpected error occurred for {episode_dir}: {e}\033[0m.")
    else:
        # This block executes if no exception occurs
        # print(f"{episode_dir} JSON files are successfully read.")
        pass


with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
    futures = [
        executor.submit(
            check_keys,
            episode_name,
            root,
            episode_config,
        )
        for episode_name in sorted(episode_names)
    ]
    for future in concurrent.futures.as_completed(futures):
        if not future.result():
            raise RuntimeError("Multi-processing failed!")

print("Finished reading.")

