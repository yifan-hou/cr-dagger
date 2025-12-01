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

output_dir = pathlib.Path(
    os.environ.get("PYRITE_DATASET_FOLDERS") + "/belt_assembly_offline_50_total_190"
)

# open the zarr store
store = zarr.DirectoryStore(path=output_dir)
root = zarr.open(store=store, mode="a")

episode_config = {
    "output_dir": output_dir,
    "id_list": id_list,
    "num_threads": 10,
    "has_correction": CORRECTION,
    "save_video": False,
    "max_workers": 32
}

print("Generating metadata")
from PyriteUtility.computer_vision.imagecodecs_numcodecs import register_codecs

register_codecs()


count = generate_meta_for_zarr(root, episode_config)
print(f"All done! Generated {count} episodes in {output_dir}")
print("The only thing left is to run postprocess_add_virtual_target_label.py")
