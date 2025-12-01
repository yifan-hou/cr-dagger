# This script does the following:
# 1. Compute the virtual target pose and stiffness based on the force/torque sensor data, add them to the zarr file
# 2. Optionally plot the virtual target and the target in 3D space
# 3. If ts_pose_command is not available, the script will populate it with ts_pose_fb.

import zarr
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

SCRIPT_PATH = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(SCRIPT_PATH, "../../"))

import concurrent.futures

from PyriteUtility.data_pipeline.processing_functions import compute_vt_for_episode

if "PYRITE_DATASET_FOLDERS" not in os.environ:
    raise ValueError("Please set the environment variable PYRITE_DATASET_FOLDERS")
dataset_folder_path = os.environ.get("PYRITE_DATASET_FOLDERS")

# Config for umift (single robot)
dataset_path = dataset_folder_path + "/online_belt_assembly_50/processed"
id_list = [0]

# # Config for vase wiping (bimanual)
# dataset_path = dataset_folder_path + "/vase_wiping_v5.2/"
# id_list = [0, 1]

# ‘r’ means read only (must exist); ‘r+’ means read/write (must exist); ‘a’ means read/write (create if doesn’t exist); ‘w’ means create (overwrite if exists); ‘w-’ means create (fail if exists).
buffer = zarr.open(dataset_path, mode="r+")

stiffness_estimation_paras = {
    "k_max": 5000,  # 1cm 50N
    "k_min": 200,  # 1cm 2.5N
    "f_low": 0.5,
    "f_high": 5,
    "dim": 3,
    "characteristic_length": 0.02,
}

vt_config = {
    "stiffness_estimation_para": stiffness_estimation_paras,
    "wrench_moving_average_window_size": 500,  # should be around 1s of data,
    "flag_real": True, # False for simulation data
    "num_of_process": 1, # 5
    "flag_plot": False,
    "fin_every_n": 50, # 50
    "id_list": id_list,
}

if vt_config["flag_plot"]:
    assert vt_config["num_of_process"] == 1, "Plotting is not supported for multi-process"


if vt_config["num_of_process"] == 1:
    for ep, ep_data in tqdm(buffer["data"].items(), desc="Episodes"):
        compute_vt_for_episode(ep, ep_data, vt_config)
else:
    with concurrent.futures.ProcessPoolExecutor(max_workers=vt_config["num_of_process"]) as executor:
        futures = [
            executor.submit(
                compute_vt_for_episode,
                ep,
                ep_data,
                vt_config,
            )
            for ep, ep_data in tqdm(buffer["data"].items(), desc="Episodes")
        ]
        for future in concurrent.futures.as_completed(futures):
            if not future.result():
                raise RuntimeError("Multi-processing failed!")


print("All Done!")
