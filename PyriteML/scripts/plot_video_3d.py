import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Cursor

import sys
import os
from typing import Dict, Callable, Tuple, List

SCRIPT_PATH = "/home/yifanhou/git/PyriteML/scripts"
sys.path.append(os.path.join(SCRIPT_PATH, '../'))

# from wrench_calibration import MLP
from PyriteUtility.spatial_math import spatial_utilities as su
from PyriteUtility.plotting.traj_and_rgb import draw_timed_traj_and_rgb
from PyriteUtility.computer_vision.imagecodecs_numcodecs import register_codecs
register_codecs()

import numpy as np
import zarr
import copy

if "PYRITE_DATASET_FOLDERS" not in os.environ:
    raise ValueError("Please set the environment variable PYRITE_DATASET_FOLDERS")
dataset_folder_path = os.environ.get("PYRITE_DATASET_FOLDERS")


show_correction = True

dataset_path = "/shared_local/data/processed/online_belt_v3/processed"
# dataset_path = "/shared_local/data/processed/belt_assembly"

print("Loading dataset from: ", dataset_path)
# load the zarr dataset from the path
# ‘r’ means read only (must exist); ‘r+’ means read/write (must exist); ‘a’ means read/write (create if doesn’t exist); ‘w’ means create (overwrite if exists); ‘w-’ means create (fail if exists).
buffer = zarr.open(dataset_path, mode="r+")

ep_count = 1
for ep, ep_data in buffer["data"].items():
    # ep = "episode_1745606000"
    # ep = "episode_1745606349"
    # ep_data = buffer["data"][ep]
    print(ep_count, ": ", ep)
    ep_count += 1
    if ep_count < 18:
        continue

    rgb_0 = buffer["data"][ep]["rgb_0"]
    rgb_time_stamps_0 = np.array(buffer["data"][ep]["rgb_time_stamps_0"])

    ts_pose_fb_0 = np.array(buffer["data"][ep]["ts_pose_fb_0"])
    robot_time_stamps_0 = np.array(buffer["data"][ep]["robot_time_stamps_0"])

    base_policy_pose_0 = np.array(buffer["data"][ep]["policy_pose_command_0"])
    base_policy_time_stamps_0 = np.array(buffer["data"][ep]["policy_time_stamps_0"])


    wrench_0 = np.array(buffer["data"][ep]["wrench_0"])
    wrench_filtered_0 = np.array(buffer["data"][ep]["wrench_filtered_0"])
    wrench_moving_average_0 = np.array(buffer["data"][ep]["wrench_moving_average_0"])
    wrench_time_stamps_0 = np.array(buffer["data"][ep]["wrench_time_stamps_0"])

    robot_wrench_0 = np.array(buffer["data"][ep]["robot_wrench_0"])
    robot_wrench_time_stamps_0 = robot_time_stamps_0

    key_event_0 = None
    key_event_timestamps_0 = None
    if show_correction:
        key_event_0 = np.array(buffer["data"][ep]["key_event_0"])
        key_event_timestamps_0 = np.array(buffer["data"][ep]["key_event_time_stamps_0"])
        


    # # load the network
    # import pickle
    # import torch
    # filename = "wrench_model_info.pkl"
    # model_name = "wrench_model.pth"
    # with open(filename, 'rb') as file:
    #     wrench_model = pickle.load(file)
    # scaler_X = wrench_model["scaler_X"]
    # scaler_Y = wrench_model["scaler_Y"]
    # input_size = wrench_model["input_size"]
    # hidden_size = wrench_model["hidden_size"]
    # output_size = wrench_model["output_size"]
    # model = MLP(input_size=input_size, hidden_size=hidden_size, output_size=output_size)
    # model.load_state_dict(torch.load(model_name, weights_only=True))
    # model.eval()
    # X_new_scaled = torch.FloatTensor(scaler_X.transform(robot_wrench_0))
    # with torch.no_grad():
    #     Y_pred_scaled = model(X_new_scaled)
    #     robot_wrench_0_updated = scaler_Y.inverse_transform(Y_pred_scaled.numpy())
    # wrench_0 = robot_wrench_0_updated - ati_0


    # # filter wrench using moving average
    # wrench_moving_average = np.zeros_like(wrench_0)
    # N = 500
    # # fmt: off
    # wrench_moving_average[:, 0] = np.convolve(wrench_0[:, 0], np.ones(N) / N, mode="same")
    # wrench_moving_average[:, 1] = np.convolve(wrench_0[:, 1], np.ones(N) / N, mode="same")
    # wrench_moving_average[:, 2] = np.convolve(wrench_0[:, 2], np.ones(N) / N, mode="same")
    # wrench_moving_average[:, 3] = np.convolve(wrench_0[:, 3], np.ones(N) / N, mode="same")
    # wrench_moving_average[:, 4] = np.convolve(wrench_0[:, 4], np.ones(N) / N, mode="same")
    # wrench_moving_average[:, 5] = np.convolve(wrench_0[:, 5], np.ones(N) / N, mode="same")
    # # fmt: on

    # convert to tip position
    SE3_Ttip = np.eye(4)
    SE3_Ttip[0:3, 3] = np.array([0.0, 0.0, 0.297])

    SE3_fb_WT = su.pose7_to_SE3(ts_pose_fb_0)
    SE3_base_policy_WT = su.pose7_to_SE3(base_policy_pose_0)

    SE3_fb_Wtip = SE3_fb_WT @ SE3_Ttip
    SE3_base_policy_Wtip = SE3_base_policy_WT @ SE3_Ttip

    ts_tip_fb_0 = su.SE3_to_pose7(SE3_fb_Wtip)
    ts_tip_base_policy_0 = su.SE3_to_pose7(SE3_base_policy_Wtip)

    data_for_plot = {
        # "ts_pose_fb_0": ts_pose_fb_0,
        # "base_policy_pose_0": base_policy_pose_0
        "ts_tip_fb_0": ts_tip_fb_0,
        "ts_tip_base_policy_0": ts_tip_base_policy_0
    }

    data_time_stamps = {
        # "ts_pose_fb_0": robot_time_stamps_0,
        # "base_policy_pose_0": base_policy_time_stamps_0
        "ts_tip_fb_0": robot_time_stamps_0,
        "ts_tip_base_policy_0": base_policy_time_stamps_0
    }

    additional_data_for_plot = {
        "wrench_0": wrench_0,
        "wrench_moving_average_0": wrench_moving_average_0,
        # "wrench_0": wrench_0[:,3:],
        # "wrench_moving_average_0": wrench_moving_average_0[:,3:],
        
        # "ati_0": ati_0,
        # "ati_adjusted": ati_adjusted,
        # "robot_wrench_0": robot_wrench_0
    }
    additional_data_time_stamps = {
        "wrench_0": wrench_time_stamps_0,
        "wrench_moving_average_0": wrench_time_stamps_0,
        # "ati_0": wrench_time_stamps_0,
        # "ati_adjusted": wrench_time_stamps_0,
        # "robot_wrench_0": wrench_time_stamps_0
    }

    draw_timed_traj_and_rgb(rgb_images = rgb_0,
        rgb_time_stamps = rgb_time_stamps_0,
        data = data_for_plot,
        data_time_stamps = data_time_stamps,
        additional_data = additional_data_for_plot,
        additional_data_time_stamps = additional_data_time_stamps,
        key_events = key_event_0,
        key_event_timestamps = key_event_timestamps_0,
        elev=20, azim=280)
