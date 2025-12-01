# This script does the following:
# 1. Compute the virtual target pose and stiffness based on the force/torque sensor data, add them to the zarr file
# 2. Optionally plot the virtual target and the target in 3D space
# 3. If ts_pose_command is not available, the script will populate it with ts_pose_fb.
# 4. offset time stamps to start from zero

import zarr
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import pdb
import cv2
import json

SCRIPT_PATH = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(SCRIPT_PATH, "../../"))

from PyriteUtility.data_pipeline.episode_data_buffer import (
    VideoData,
    EpisodeDataBuffer,
    EpisodeDataIncreImageBuffer,
)
from spatialmath.base import q2r, r2q
from spatialmath import SE3, SO3, UnitQuaternion
import concurrent.futures

from PyriteUtility.planning_control import compliance_helpers as ch
from PyriteUtility.spatial_math import spatial_utilities as su
from PyriteUtility.plotting.matplotlib_helpers import set_axes_equal

if "PYRITE_DATASET_FOLDERS" not in os.environ:
    raise ValueError("Please set the environment variable PYRITE_DATASET_FOLDERS")
dataset_folder_path = os.environ.get("PYRITE_DATASET_FOLDERS")

# Config for umift (single robot)
dataset_path = '/store/real/hjchoi92/data/real_processed/umift/WBW-iph/acp_replay_buffer_gripper.zarr' # IMPORTANT: Make sure to change the mute flags too! -HC
# dataset_path = dataset_folder_path + "/umift/acp_replay_buffer_gripper.zarr/"
id_list = [0]

wrench_moving_average_window_size = 200  # should be around 1s of data

# ‘r’ means read only (must exist); ‘r+’ means read/write (must exist); ‘a’ means read/write (create if doesn’t exist); ‘w’ means create (overwrite if exists); ‘w-’ means create (fail if exists).
buffer = zarr.open(dataset_path, mode="r+")


num_of_process = 1  # 5
flag_plot = False
fin_every_n = 25  # 50

depth_clip = 0.5

apply_mask = True
clip_depth = True
convert_BGR_to_RGB = False
flag_mute_gripper = False
flag_mute_wrench = False


print(f"Applying mask to the dataset")

mask_path = os.path.join(SCRIPT_PATH, "umift_mask.json")
with open(mask_path, "r") as f:
    mask_data = json.load(f)

mask_pts = np.array(mask_data["umift_uw_mask"], dtype=np.int32)  # Extract mask points
mask_resolution = tuple(mask_data["mask_uw_resolution"])  # (W, H)

print(f"Mask resolution: {mask_resolution}")

if flag_mute_gripper:
    # there is no point of having wrench if the gripper is muted.
    flag_mute_wrench = True

print(f"[Modality Ablation] Is the gripper width information being muted: {flag_mute_gripper}")
print(f"[Modality Ablation] Is the wrench information being muted: {flag_mute_wrench}")


stiffness_estimation_para = {
    # penetration estimator
    "k_max": 3000,  # 1cm 50N
    "k_min": 200,  # 1cm 2.5N
    "f_low": 2.85, #0.5,
    # "f_low": 1.5,
    "f_high": 7, #5,
    "dim": 3,
    "characteristic_length": 0.02,
    "vel_tol": 999.002,  # vel larger than this will trigger stiffness adjustment
}

flag_real = False
if "real" in dataset_path:
    flag_real = True
    print("flag_real set!")

if flag_plot:
    assert num_of_process == 1, "Plotting is not supported for multi-process"


def process_episode(ep, ep_data, id_list):
    for key in ep_data.keys():
        print(key)

    # # HC: TODO Debugging
    # if f"robot_time_stamps_0" not in ep_data.keys():
    #     print("Robot time stamp not found in episode: ", ep)
    #     return

    # if f"robot_time_stamps_0" in ep_data.keys():
    #     print("Of course you see this in episode:", ep)


    for id in id_list:
        print(f"Processing episode {ep}, id {id}: ")

        if flag_mute_gripper:
            ep_data[f"gripper_{id}"][:] = 0
        
        if flag_mute_wrench:
            ep_data[f"wrench_concat_{id}"][:] = 0
            ep_data[f"wrench_concat_coinft_{id}"][:] = 0
            ep_data[f"wrench_left_{id}"][:] = 0
            ep_data[f"wrench_left_coinft_{id}"][:] = 0
            ep_data[f"wrench_right_{id}"][:] = 0
            ep_data[f"wrench_right_coinft_{id}"][:] = 0

        # if RGB processing has not been done on this episode, do it
        if convert_BGR_to_RGB:
            rgb_conversion_flag = f"rgb_converted_{id}"
            if rgb_conversion_flag in ep_data.keys():
                print(f"Skipping RGB conversion for episode {ep}, id {id} (already processed)")
            else:
                print(f"Converting BGR to RGB for episode {ep}, id {id}")
                rgb_images = np.array(ep_data[f"rgb_0"])
                rgb_images = np.array([cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in rgb_images])
                ep_data[f"rgb_0"] = rgb_images
                ep_data[rgb_conversion_flag] = np.array([1])
        
        if apply_mask:
            images = np.array(ep_data[f"ultrawide_0"])
            # create a mask where the masked area is 0 and others are all 255 (0xFF)
            mask = np.ones_like(images[0], dtype=np.uint8) * 255
            cv2.fillPoly(mask, [mask_pts], (0, 0, 0))

            # apply the mask to the images
            images = np.array([cv2.bitwise_and(img, mask) for img in images])

            ep_data[f"ultrawide_0"] = images

        if clip_depth:
            depth_images = np.array(ep_data[f"depth_0"])
            depth_images_clipped = np.clip(depth_images, a_min=0.0, a_max=depth_clip)
            ep_data[f"depth_0"] = depth_images_clipped


        ts_pose_fb = ep_data[f"ts_pose_fb_{id}"]
        wrench_left = np.array(ep_data[f"wrench_left_{id}"])
        wrench_right = np.array(ep_data[f"wrench_right_{id}"])

        wrench = wrench_left + wrench_right

        # make time stamps start from zero

        time_offsets = []
        for id in id_list:
            time_offsets.append(ep_data[f"rgb_time_stamps_{id}"][0])
            time_offsets.append(ep_data[f"ultrawide_time_stamps_{id}"][0])
            time_offsets.append(ep_data[f"depth_time_stamps_{id}"][0])
            time_offsets.append(ep_data[f"robot_time_stamps_{id}"][0])
            time_offsets.append(ep_data[f"wrench_time_stamps_left_{id}"][0]) 
            time_offsets.append(ep_data[f"wrench_time_stamps_right_{id}"][0]) 
        print("Time offsets: ", time_offsets)
        time_offset = np.min(time_offsets)
        for id in id_list:
            ep_data[f"wrench_time_stamps_left_{id}"] -= time_offset 
            ep_data[f"wrench_time_stamps_right_{id}"] -= time_offset 
            ep_data[f"robot_time_stamps_{id}"] -= time_offset
            ep_data[f"rgb_time_stamps_{id}"] -= time_offset
            ep_data[f"ultrawide_time_stamps_{id}"] -= time_offset
            ep_data[f"depth_time_stamps_{id}"] -= time_offset
            if f"gripper_time_stamps_{id}" in ep_data.keys():
                ep_data[f"gripper_time_stamps_{id}"] -= time_offset

        wrench_time_stamps = (np.array(ep_data[f"wrench_time_stamps_left_{id}"]) + np.array(ep_data[f"wrench_time_stamps_right_{id}"])) / 2
        robot_time_stamps = ep_data[f"robot_time_stamps_{id}"]
        rgb_time_stamps = ep_data[f"rgb_time_stamps_{id}"]
        wrench_moving_average = np.zeros_like(wrench)


        # remove wrench measurement offset
        Noffset = 200
        wrench_offset = np.mean(wrench[:Noffset], axis=0)
        print("wrench offset: ", wrench_offset)

        # # FT300 only: flip the sign of the wrench
        # for i in range(6):
        #     wrench[:, i] = -wrench[:, i]

        # filter wrench using moving average
        N = wrench_moving_average_window_size
        print("Computing moving average")
        # fmt: off
        wrench_moving_average[:, 0] = np.convolve(wrench[:, 0], np.ones(N) / N, mode="same")
        wrench_moving_average[:, 1] = np.convolve(wrench[:, 1], np.ones(N) / N, mode="same")
        wrench_moving_average[:, 2] = np.convolve(wrench[:, 2], np.ones(N) / N, mode="same")
        wrench_moving_average[:, 3] = np.convolve(wrench[:, 3], np.ones(N) / N, mode="same")
        wrench_moving_average[:, 4] = np.convolve(wrench[:, 4], np.ones(N) / N, mode="same")
        wrench_moving_average[:, 5] = np.convolve(wrench[:, 5], np.ones(N) / N, mode="same")
        # fmt: on

        if not flag_real:  # for simulation data
            ft_sensor_pose_fb = ep_data["ft_sensor_pose_fb"]

        num_robot_time_steps = len(robot_time_stamps)

        print("creating virtual target estimator")

        pe = ch.VirtualTargetEstimator(
            stiffness_estimation_para["k_max"],
            stiffness_estimation_para["k_min"],
            stiffness_estimation_para["f_low"],
            stiffness_estimation_para["f_high"],
            stiffness_estimation_para["dim"],
            stiffness_estimation_para["characteristic_length"],
        )

        ts_pose_virtual_target = np.zeros((num_robot_time_steps, 7))
        stiffness = np.zeros(num_robot_time_steps)

        print("Running virtual target estimator")
        pose7_WT = ts_pose_fb
        SE3_WT = su.pose7_to_SE3(pose7_WT)

        # # find the id in wrench_time_stamps where the time is closest to robot_time_stamps
        # print("wrench_time_stamps: ", wrench_time_stamps.shape)
        # print("robot_time_stamps: ", robot_time_stamps.shape)
        wrench_id = np.searchsorted(np.squeeze(wrench_time_stamps), np.squeeze(robot_time_stamps))
        wrench_id = np.minimum(wrench_id, len(wrench_time_stamps) - 1)
        if flag_real:
            wrench_T = wrench_moving_average[wrench_id]
        else:
            assert False, "Not implemented. Need to parallelize this part"

        # compute stiffness
        if stiffness_estimation_para["dim"] == 6:
            k, SE3_TC = pe.batch_update(wrench_T)
        else:
            k, pos_TC = pe.batch_update(wrench_T)
            SE3_TC = np.zeros((num_robot_time_steps, 4, 4))
            SE3_TC[:, :4, :4] = np.eye(4)
            SE3_TC[:, :3, 3] = pos_TC
        SE3_WC = SE3_WT @ SE3_TC
        
        ts_pose_virtual_target = su.SE3_to_pose7(SE3_WC)
        stiffness = np.squeeze(k)

        ep_data[f"ts_pose_virtual_target_{id}"] = ts_pose_virtual_target
        ep_data[f"stiffness_{id}"] = stiffness
        ep_data[f"wrench_time_stamps_{id}"] = wrench_time_stamps

        if f"ts_pose_command_{id}" not in ep_data:
            ep_data[f"ts_pose_command_{id}"] = ts_pose_fb

        print("Done")

        if flag_plot:
            print("Plotting...")
            plt.ion()  # to run GUI event loop
            fig = plt.figure()
            ax = plt.axes(projection="3d")
            x = np.linspace(-0.02, 0.2, 20)
            y = np.linspace(-0.1, 0.1, 20)
            z = np.linspace(-0.1, 0.1, 20)
            ax.plot3D(x, y, z, color="blue", marker="o", markersize=3)
            ax.plot3D(x, y, z, color="red", marker="o", markersize=3)
            ax.set_title("Target and virtual target")
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            plt.show()

            ax.cla()
            ax.plot3D(
                ts_pose_fb[..., 0],
                ts_pose_fb[..., 1],
                ts_pose_fb[..., 2],
                color="red",
                marker="o",
                markersize=2,
            )

            ax.plot3D(
                ts_pose_virtual_target[..., 0],
                ts_pose_virtual_target[..., 1],
                ts_pose_virtual_target[..., 2],
                color="blue",
                marker="o",
                markersize=2,
            )
            # starting point
            ax.plot3D(
                ts_pose_fb[0][0],
                ts_pose_fb[0][1],
                ts_pose_fb[0][2],
                color="black",
                marker="o",
                markersize=8,
            )

            # fin
            for i in np.arange(0, num_robot_time_steps, fin_every_n):
                ax.plot3D(
                    [ts_pose_fb[i][0], ts_pose_virtual_target[i][0]],
                    [ts_pose_fb[i][1], ts_pose_virtual_target[i][1]],
                    [ts_pose_fb[i][2], ts_pose_virtual_target[i][2]],
                    color="black",
                    marker="o",
                    markersize=2,
                )

            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")

            set_axes_equal(ax)

            plt.draw()
            plt.savefig(f"plot_{ep}.png")

            input("Press Enter to continue...")
        return True


# selected_data = {}
# count = 0
# loop_count = 0
# for key, value in buffer["data"].items():
#     loop_count += 1
#     if loop_count % 7 == 0:
#         selected_data[key] = value
#         print("Selected data: ", key)
#         count += 1
#         if count == 25:
#             break

print("num_of_process: ", num_of_process)
if num_of_process == 1:
    # for ep, ep_data in tqdm(selected_data.items(), desc="Episodes"):
    for ep, ep_data in tqdm(buffer["data"].items(), desc="Episodes"):
        print("ep name: ", ep)
        # pdb.set_trace()
        process_episode(ep, ep_data, id_list)
else:
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_of_process) as executor:
        futures = [
            executor.submit(
                process_episode,
                ep,
                ep_data,
                id_list,
            )
            # for ep, ep_data in tqdm(selected_data.items(), desc="Episodes") ]
            for ep, ep_data in tqdm(buffer["data"].items(), desc="Episodes") ]
        for future in concurrent.futures.as_completed(futures):
            if not future.result():
                raise RuntimeError("Multi-processing failed!")


print("All Done!")