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


# specify the input and output director
id_list = [0]  # single robot
# id_list = [0, 1] # bimanual

ft_sensor_configuration = "handle_on_robot" # "handle_on_sensor" or "handle_on_robot"

input_dir = pathlib.Path(
    os.environ.get("PYRITE_RAW_DATASET_FOLDERS") + "/belt_assembly_50"
)
# input_dir = pathlib.Path(
#     os.environ.get("PYRITE_DATASET_FOLDERS") + "/online_stow_nb_v5_50/raw"
# )
output_dir = pathlib.Path(
    os.environ.get("PYRITE_DATASET_FOLDERS") + "/belt_assembly_offline_50"
)

# clean and create output folders
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)

# # check for black images
# def check_black_images(rgb_file_list, rgb_dir, i, prefix):
#     f = rgb_file_list[i]
#     img = cv2.imread(str(rgb_dir.joinpath(f)))
#     # print the mean of the image
#     img_mean = np.mean(img)
#     if img_mean < 50:
#         print(f"{prefix}, {f} has mean value of {img_mean}")
#     return True


# for episode_name in os.listdir(input_dir):
#     if episode_name.startswith("."):
#         continue

#     episode_dir = input_dir.joinpath(episode_name)
#     for id in id_list:
#         rgb_dir = episode_dir.joinpath("rgb_" + str(id))
#         rgb_file_list = os.listdir(rgb_dir)
#         num_raw_images = len(rgb_file_list)
#         print(f"Checking for black images in {rgb_dir}")
#         with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
#             futures = set()
#             for i in range(len(rgb_file_list)):
#                 futures.add(
#                     executor.submit(
#                         check_black_images,
#                         rgb_file_list,
#                         rgb_dir,
#                         i,
#                         f"{episode_name} rgb_{id}",
#                     )
#                 )

#             completed, futures = concurrent.futures.wait(futures)
#             for f in completed:
#                 if not f.result():
#                     raise RuntimeError("Failed to read image!")
# exit()

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
    "num_threads": 10,
    "has_correction": CORRECTION,
    "save_video": False,
    "max_workers": 32
}

with concurrent.futures.ProcessPoolExecutor(max_workers=3) as executor:
    futures = [
        executor.submit(
            process_one_episode_into_zarr,
            episode_name,
            root,
            episode_config,
        )
        for episode_name in episode_names
    ]
    for future in concurrent.futures.as_completed(futures):
        if not future.result():
            raise RuntimeError("Multi-processing failed!")

print("Finished reading. Now start generating metadata")
from PyriteUtility.computer_vision.imagecodecs_numcodecs import register_codecs

register_codecs()


count = generate_meta_for_zarr(root, episode_config)
print(f"All done! Generated {count} episodes in {output_dir}")
print("The only thing left is to run postprocess_add_virtual_target_label.py")
