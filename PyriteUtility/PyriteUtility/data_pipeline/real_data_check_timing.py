import sys
import os

SCRIPT_PATH = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(SCRIPT_PATH, "../../"))

from PyriteUtility.data_pipeline.processing_functions import process_one_episode_into_zarr, generate_meta_for_zarr

import pathlib
import shutil
import numpy as np
import zarr
import cv2
import concurrent.futures

CORRECTION = False   # set to true if you want to use the correction data

# check environment variables
if "PYRITE_RAW_DATASET_FOLDERS" not in os.environ:
    raise ValueError("Please set the environment variable PYRITE_RAW_DATASET_FOLDERS")
if "PYRITE_DATASET_FOLDERS" not in os.environ:
    raise ValueError("Please set the environment variable PYRITE_DATASET_FOLDERS")


# specify the input and output directories
id_list = [0]  # single robot
# id_list = [0, 1] # bimanual

ft_sensor_configuration = "handle_on_robot" # "handle_on_sensor" or "handle_on_robot"

input_dir = pathlib.Path(
os.environ.get("PYRITE_RAW_DATASET_FOLDERS") + "/belt_assembly_v1"
)
output_dir = pathlib.Path(
os.environ.get("PYRITE_DATASET_FOLDERS") + "/belt_assembly_v1"
)

# open the zarr store
store = zarr.DirectoryStore(path=output_dir)
root = zarr.open(store=store, mode="a")

print("Reading data from input_dir: ", input_dir)
episode_names = os.listdir(input_dir)

episode_config = {
    "input_dir": input_dir,
    "output_dir": output_dir,
    "id_list": id_list,
    "ft_sensor_configuration": ft_sensor_configuration,
    "num_threads": 1,
    "has_correction": CORRECTION,
    "save_video": False,
    "max_workers": 32
}


import pandas as pd

def check_timing(episode_name, output_root, config):
    if episode_name.startswith("."):
        return True

    # info about input
    episode_id = episode_name[8:]
    # print(f"[process_one_episode_into_zarr] episode_name: {episode_name}, episode_id: {episode_id}")
    episode_dir = pathlib.Path(config["input_dir"]).joinpath(episode_name)

    # read low dim data
    data_ts_pose_fb = []
    data_robot_time_stamps = []
    data_robot_wrench = []
    data_wrench = []
    data_wrench_filtered = []
    data_wrench_time_stamps = []
    data_masks = []
    # print(f"Reading low dim data for : {episode_dir}")
    try:
        for id in config["id_list"]:
            # read robot data
            json_path = episode_dir.joinpath("robot_data_" + str(id) + ".json")
            df_robot_data = pd.read_json(json_path)
            robot_time_stamps = df_robot_data["robot_time_stamps"].to_numpy()
            
            delta_time = robot_time_stamps[1:] - robot_time_stamps[:-1]
            average_delta_time = np.mean(delta_time)
            print(f"Average delta time for {episode_name}: {average_delta_time}")
            

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
            check_timing,
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

