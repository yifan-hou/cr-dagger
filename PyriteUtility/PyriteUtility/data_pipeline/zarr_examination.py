import zarr
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

SCRIPT_PATH = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(SCRIPT_PATH, "../../"))

if "PYRITE_DATASET_FOLDERS" not in os.environ:
    raise ValueError("Please set the environment variable PYRITE_DATASET_FOLDERS")
dataset_folder_path = os.environ.get("PYRITE_DATASET_FOLDERS")


from PyriteUtility.computer_vision.imagecodecs_numcodecs import (
    register_codecs,
    Jpeg2k,
    JpegXl,
)

register_codecs()



# Config for umift (single robot)
dataset_path = dataset_folder_path + "/online_belt_assembly_50/processed"
id_list = [0]

# ‘r’ means read only (must exist); ‘r+’ means read/write (must exist); ‘a’ means read/write (create if doesn’t exist); ‘w’ means create (overwrite if exists); ‘w-’ means create (fail if exists).
buffer = zarr.open(dataset_path, mode="r+")

for ep, ep_data in buffer["data"].items():
    for key in ep_data.keys():
        print(f"key: {key}, shape: {ep_data[key].shape}, dtype: {ep_data[key].dtype}")
    break  # Remove this line to iterate through all episodes

print("All Done!")
