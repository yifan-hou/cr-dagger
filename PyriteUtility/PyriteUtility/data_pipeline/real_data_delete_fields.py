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

# check environment variables
if "PYRITE_RAW_DATASET_FOLDERS" not in os.environ:
    raise ValueError("Please set the environment variable PYRITE_RAW_DATASET_FOLDERS")
if "PYRITE_DATASET_FOLDERS" not in os.environ:
    raise ValueError("Please set the environment variable PYRITE_DATASET_FOLDERS")


# specify the input and output directories
id_list = [0]  # single robot
# id_list = [0, 1] # bimanual

output_dir = pathlib.Path(
    os.environ.get("PYRITE_DATASET_FOLDERS") + "/online_stow_nb_v5_50/processed_no_correction/data"
)

# clean and create output folders

files_to_delete = []

for episode_name in os.listdir(output_dir):
    if episode_name.startswith("."):
        continue

    episode_dir = output_dir.joinpath(episode_name)
    print("Checking episode: ", episode_dir)
    for id in id_list:
        files_to_delete.append(episode_dir.joinpath("policy_pose_command_" + str(id)))
        files_to_delete.append(episode_dir.joinpath("policy_time_stamps_" + str(id)))

for file in files_to_delete:
    print("Deleting files: ", file)
input("Press Enter to continue...")
for file in files_to_delete:
    if os.path.exists(file):
        shutil.rmtree(file)
print("All done! Deleted files")