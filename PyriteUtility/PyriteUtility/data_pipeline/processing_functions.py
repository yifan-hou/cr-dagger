import sys
import os

SCRIPT_PATH = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(SCRIPT_PATH, "../../"))

from PyriteUtility.data_pipeline.episode_data_buffer import (
    VideoData,
    EpisodeDataBuffer,
)

import numpy as np
import pandas as pd
import zarr
import cv2
import concurrent.futures
import pathlib
import tqdm

import matplotlib.pyplot as plt

from spatialmath.base import q2r, r2q
from spatialmath import SE3, SO3, UnitQuaternion
import concurrent.futures

from PyriteUtility.planning_control import compliance_helpers as ch
from PyriteUtility.plotting.matplotlib_helpers import set_axes_equal
from PyriteUtility.spatial_math import spatial_utilities as su


def image_read(rgb_dir, rgb_file_list, i, output_data_rgb, output_data_rgb_time_stamps):
    img_name = rgb_file_list[i]
    img = cv2.imread(str(rgb_dir.joinpath(img_name)))
    # convert BGR to RGB for imageio
    output_data_rgb[i] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    time_img_ms = float(img_name[11:22])
    output_data_rgb_time_stamps[i] = time_img_ms
    return True

### Process one episode:
### 1. Read raw data on file from episode_name, input_dir
### 2. Save the episode data to output_store
### 3. Save a video of the episode to output_dir
def process_one_episode_into_zarr(episode_name, output_root, config):
    if episode_name.startswith("."):
        return True

    # info about input
    episode_id = episode_name[8:]
    print(f"[process_one_episode_into_zarr] episode_name: {episode_name}, episode_id: {episode_id}")
    episode_dir = pathlib.Path(config["input_dir"]).joinpath(episode_name)

    # read rgb
    data_rgb = []
    data_rgb_time_stamps = []
    rgb_data_shapes = []
    for id in config["id_list"]:
        rgb_dir = episode_dir.joinpath("rgb_" + str(id))
        rgb_file_list = os.listdir(rgb_dir)
        if len(rgb_file_list) > 0:
            rgb_file_list.sort()  # important!
            num_raw_images = len(rgb_file_list)
            img = cv2.imread(str(rgb_dir.joinpath(rgb_file_list[0])))

            rgb_data_shapes.append((num_raw_images, *img.shape))
            data_rgb.append(np.zeros(rgb_data_shapes[id], dtype=np.uint8))
            data_rgb_time_stamps.append(np.zeros(num_raw_images))

            print(f"Reading rgb data from: {rgb_dir}")
            with concurrent.futures.ThreadPoolExecutor(max_workers=config["num_threads"]) as executor:
                futures = set()
                for i in range(len(rgb_file_list)):
                    futures.add(
                        executor.submit(
                            image_read,
                            rgb_dir,
                            rgb_file_list,
                            i,
                            data_rgb[id],
                            data_rgb_time_stamps[id],
                        )
                    )

                completed, futures = concurrent.futures.wait(futures)
                for f in completed:
                    if not f.result():
                        raise RuntimeError("Failed to read image!")

    # read low dim data
    data_ts_pose_fb = []
    data_robot_time_stamps = []
    data_robot_wrench = []
    data_wrench = []
    data_wrench_filtered = []
    data_wrench_time_stamps = []
    data_masks = []
    data_key = []
    data_key_time_stamps = []
    data_gripper = []
    data_gripper_time_stamps = []

    print(f"Reading low dim data for : {episode_dir}")
    try:
        for id in config["id_list"]:
            # read robot data
            json_path = episode_dir.joinpath("robot_data_" + str(id) + ".json")
            df_robot_data = pd.read_json(json_path)
            data_robot_time_stamps.append(df_robot_data["robot_time_stamps"].to_numpy())
            data_ts_pose_fb.append(np.vstack(df_robot_data["ts_pose_fb"]))
            data_robot_wrench.append(np.vstack(df_robot_data["robot_wrench"]))
            data_masks.append(np.vstack(df_robot_data["mask"]))

            # read wrench data
            json_path = episode_dir.joinpath("wrench_data_" + str(id) + ".json")
            if os.path.exists(json_path):
                df_wrench_data = pd.read_json(json_path)
                data_wrench_time_stamps.append(df_wrench_data["wrench_time_stamps"].to_numpy())
                data_wrench.append(np.vstack(df_wrench_data["wrench"]))
                data_wrench_filtered.append(np.vstack(df_wrench_data["wrench_filtered"]))

            # read gripper
            json_path = episode_dir.joinpath("eoat_data_" + str(id) + ".json")
            if os.path.exists(json_path):
                df_gripper_data = pd.read_json(json_path)
                data_gripper.append(np.vstack(df_gripper_data["eoat_pos_fb"]))
                data_gripper_time_stamps.append(df_gripper_data["eoat_time_stamps"].to_numpy())

            if config["ft_sensor_configuration"] == "handle_on_sensor":
                # Compute wrench_net = robot_wrench - wrench, where we use the robot_wrench that is closest to the wrench time stamp
                robot_wrench_id = np.searchsorted(data_robot_time_stamps[id], data_wrench_time_stamps[id])
                Nrobot = len(data_robot_time_stamps[id])
                robot_wrench_id = np.minimum(robot_wrench_id, Nrobot - 1)
                wrench_net = data_robot_wrench[id][robot_wrench_id] - data_wrench[id]
                wrench_filtered_net = data_robot_wrench[id][robot_wrench_id] - data_wrench_filtered[id]
                data_wrench[id] = wrench_net
                data_wrench_filtered[id] = wrench_filtered_net
                data_robot_wrench[id] = data_robot_wrench[id][robot_wrench_id]

            # read key data
            json_path = episode_dir.joinpath("key_data.json")
            if os.path.exists(json_path):
                df_key_data = pd.read_json(json_path)
                data_key_time_stamps.append(df_key_data["key_event_time_stamps"].to_numpy())
                data_key.append(np.vstack(df_key_data["key_event"]))
        # read correction data (policy inference data)
        if config["has_correction"]:
            policy_inference_path = episode_dir.joinpath("policy_inference.zarr")
            if os.path.exists(policy_inference_path):
                policy_data = zarr.open(policy_inference_path, mode='r')
                # print the shapes of attributes in policy_data
                for k in policy_data.keys():
                    print(f"{k}: {policy_data[k].shape}")
                data_policy_ts_pose_command = []
                data_policy_time_stamps = []
                data_policy_gripper_command = []
                for id in config["id_list"]:
                    data_policy_ts_pose_command.append(policy_data[f'ts_targets_{id}'][:].reshape(-1, 7))
                    data_policy_time_stamps.append(policy_data['timestamps_s'][:].reshape(-1)*1000.0)
                    if f'ts_gripper_{id}' in policy_data:
                        data_policy_gripper_command.append(policy_data[f'ts_grippers_{id}'][:].reshape(-1, 1))
                print(f"Correction data found in {policy_inference_path}")
            else:
                print(f"Correction data NOT found in {policy_inference_path}")
                exit()
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
        print(f"{episode_dir} JSON files are successfully read.")

    # make time stamps start from zero
    time_offsets = []
    for id in config["id_list"]:
        time_offsets.append(data_robot_time_stamps[id][0])
        if len(data_rgb_time_stamps) > 0:      
            time_offsets.append(data_rgb_time_stamps[id][0])
        if len(data_wrench_time_stamps) > 0:
            time_offsets.append(data_wrench_time_stamps[id][0])
        if config["has_correction"]:
            time_offsets.append(data_policy_time_stamps[id][0])
        if len(data_key_time_stamps) > 0:
            time_offsets.append(data_key_time_stamps[id][0])
    time_offset = np.min(time_offsets)
    for id in config["id_list"]:
        data_robot_time_stamps[id] -= time_offset
        if len(data_rgb_time_stamps) > 0:
            data_rgb_time_stamps[id] -= time_offset
        if len(data_wrench_time_stamps) > 0:
            data_wrench_time_stamps[id] -= time_offset
        if config["has_correction"]:
            data_policy_time_stamps[id] -= time_offset
        if len(data_key_time_stamps) > 0:
            data_key_time_stamps[id] -= time_offset

    # create output zarr
    print(f"Saving everything to : {config['output_dir']}")
    recoder_buffer = EpisodeDataBuffer(
        store_path=config["output_dir"],
        camera_ids=config["id_list"],
        save_video=config["save_video"],
        save_video_fps=60,
        data=output_root,
        max_workers=config["max_workers"],
    )

    # save data using recoder_buffer
    recoder_buffer.create_zarr_groups_for_episode(rgb_data_shapes, config["id_list"], episode_id)

    if len(data_rgb) > 0:
        rgb_data_buffer = {}
        for id in config["id_list"]:
            rgb_data = data_rgb[id]
            rgb_data_buffer.update({id: VideoData(rgb=rgb_data, camera_id=id)})
        recoder_buffer.save_video_for_episode(
            visual_observations=rgb_data_buffer,
            visual_time_stamps=data_rgb_time_stamps,
            episode_id=episode_id,
        )

    if config["has_correction"]:
        recoder_buffer.save_low_dim_for_episode(
            ts_pose_command=data_ts_pose_fb,
            ts_pose_fb=data_ts_pose_fb,
            policy_pose_command=data_policy_ts_pose_command,
            policy_gripper_command=data_policy_gripper_command,
            robot_wrench=data_robot_wrench,
            wrench=data_wrench,
            wrench_filtered=data_wrench_filtered,
            gripper_fb=data_gripper,
            key_event=data_key,
            key_event_time_stamps=data_key_time_stamps,
            robot_time_stamps=data_robot_time_stamps,
            wrench_time_stamps=data_wrench_time_stamps,
            policy_time_stamps=data_policy_time_stamps,
            gripper_time_stamps=data_gripper_time_stamps,
            episode_id=episode_id,
            masks=data_masks,
        )
    else:
        recoder_buffer.save_low_dim_for_episode(
            ts_pose_command=data_ts_pose_fb,
            ts_pose_fb=data_ts_pose_fb,
            robot_wrench=data_robot_wrench,
            wrench=data_wrench,
            wrench_filtered=data_wrench_filtered,
            gripper_fb=data_gripper,
            key_event=data_key,
            key_event_time_stamps=data_key_time_stamps,
            robot_time_stamps=data_robot_time_stamps,
            wrench_time_stamps=data_wrench_time_stamps,
            gripper_time_stamps=data_gripper_time_stamps,
            episode_id=episode_id,
            masks=data_masks,
        )
    
    return True

def generate_meta_for_zarr(root, config):
    meta = root.create_group("meta", overwrite=True)
    episode_robot_len = []
    episode_wrench_len = []
    episode_rgb_len = []
    if config["has_correction"]:
        episode_policy_len = []
    episode_key_len = []

    for id in config["id_list"]:
        episode_robot_len.append([])
        episode_wrench_len.append([])
        episode_rgb_len.append([])
        if config["has_correction"]:
            episode_policy_len.append([])
        episode_key_len.append([])

    count = 0
    for key in root["data"].keys():
        episode = key
        ep_data = root["data"][episode]

        for id in config["id_list"]:
            robot_len = 0
            wrench_len = 0
            rgb_len = 0
            policy_len = 0

            if f"ts_pose_fb_{id}" in ep_data:
                episode_robot_len[id].append(ep_data[f"ts_pose_fb_{id}"].shape[0])
            else:
                assert f"js_fb_{id}" in ep_data, f"Neither ts_pose_fb_{id} nor js_fb_{id} found in episode {episode}"
                episode_robot_len[id].append(ep_data[f"js_fb_{id}"].shape[0])
            robot_len = episode_robot_len[id][-1]
            if f"wrench_{id}" in ep_data:
                episode_wrench_len[id].append(ep_data[f"wrench_{id}"].shape[0])
                wrench_len = episode_wrench_len[id][-1]
            
            if f"rgb_{id}" in ep_data:
                episode_rgb_len[id].append(ep_data[f"rgb_{id}"].shape[0])
                rgb_len = episode_rgb_len[id][-1]

            if config["has_correction"]:
                episode_policy_len[id].append(ep_data[f"policy_pose_command_{id}"].shape[0])
                policy_len = episode_policy_len[id][-1]
                print(
                    f"Number {count}: {episode}: id = {id}: robot len: {robot_len}, wrench_len: {wrench_len} rgb len: {rgb_len} policy len: {policy_len}"
                )
            else:
                print(
                    f"Number {count}: {episode}: id = {id}: robot len: {robot_len}, wrench_len: {wrench_len} rgb len: {rgb_len}"
                )
            if f"key_event_{id}" in ep_data:
                episode_key_len[id].append(ep_data[f"key_event_{id}"].shape[0])
                key_len = episode_key_len[id][-1]
                print(f"Number {count}: {episode}: id = {id}: key len: {key_len}")
        count += 1

    for id in config["id_list"]:
        meta[f"episode_robot{id}_len"] = zarr.array(episode_robot_len[id])
        if len(episode_wrench_len[id]) > 0:
            meta[f"episode_wrench{id}_len"] = zarr.array(episode_wrench_len[id])
        if len(episode_rgb_len[id]) > 0:
            meta[f"episode_rgb{id}_len"] = zarr.array(episode_rgb_len[id])
        if config["has_correction"]:
            meta[f"episode_policy{id}_len"] = zarr.array(episode_policy_len[id])
        if len(episode_key_len[id]) > 0:
            meta[f"episode_key{id}_len"] = zarr.array(episode_key_len[id])

    # debug info
    print("episode_robot_len average: ", np.mean(np.array(episode_robot_len), axis=1))
    return count

def compute_vt_for_episode(ep, ep_data, config):
    for key in ep_data.keys():
        print(key)

    for id in config["id_list"]:
        print(f"Processing episode {ep}, id {id}: ")
        ts_pose_fb = ep_data[f"ts_pose_fb_{id}"]
        wrench = ep_data[f"wrench_{id}"]

        wrench_time_stamps = ep_data[f"wrench_time_stamps_{id}"]
        robot_time_stamps = ep_data[f"robot_time_stamps_{id}"]
        
        wrench_moving_average = np.zeros_like(wrench)

        # remove wrench measurement offset
        Noffset = 200
        wrench_offset = np.mean(wrench[:Noffset], axis=0)
        print("wrench offset: ", wrench_offset)

        # # FT300 only: flip the sign of the wrench
        # for i in range(6):
        #     wrench[:, i] = -wrench[:, i]

        # filter wrench using moving average
        N = config["wrench_moving_average_window_size"]
        print("Computing moving average")
        # fmt: off
        wrench_moving_average[:, 0] = np.convolve(wrench[:, 0], np.ones(N) / N, mode="same")
        wrench_moving_average[:, 1] = np.convolve(wrench[:, 1], np.ones(N) / N, mode="same")
        wrench_moving_average[:, 2] = np.convolve(wrench[:, 2], np.ones(N) / N, mode="same")
        wrench_moving_average[:, 3] = np.convolve(wrench[:, 3], np.ones(N) / N, mode="same")
        wrench_moving_average[:, 4] = np.convolve(wrench[:, 4], np.ones(N) / N, mode="same")
        wrench_moving_average[:, 5] = np.convolve(wrench[:, 5], np.ones(N) / N, mode="same")
        # fmt: on

        if not config["flag_real"]:  # for simulation data
            ft_sensor_pose_fb = ep_data["ft_sensor_pose_fb"]

        num_robot_time_steps = len(robot_time_stamps)

        print("creating virtual target estimator")

        pe = ch.VirtualTargetEstimator(
            config["stiffness_estimation_para"]["k_max"],
            config["stiffness_estimation_para"]["k_min"],
            config["stiffness_estimation_para"]["f_low"],
            config["stiffness_estimation_para"]["f_high"],
            config["stiffness_estimation_para"]["dim"],
            config["stiffness_estimation_para"]["characteristic_length"],
        )

        ts_pose_virtual_target = np.zeros((num_robot_time_steps, 7))
        stiffness = np.zeros(num_robot_time_steps)

        print("Running virtual target estimator (new, parallel)")
        pose7_WT = ts_pose_fb
        SE3_WT = su.pose7_to_SE3(pose7_WT)

        # find the id in wrench_time_stamps where the time is closest to robot_time_stamps
        wrench_id = np.searchsorted(wrench_time_stamps, robot_time_stamps)
        wrench_id = np.minimum(wrench_id, len(wrench_time_stamps) - 1)

        if config["flag_real"]:
            wrench_T = wrench_moving_average[wrench_id]
        else:
            assert False, "Not implemented. Need to parallelize this part"

        # compute stiffness
        if config["stiffness_estimation_para"]["dim"] == 6:
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
        ep_data[f"wrench_moving_average_{id}"] = wrench_moving_average

        if f"ts_pose_command_{id}" not in ep_data:
            ep_data[f"ts_pose_command_{id}"] = ts_pose_fb

        print("Done")

        if config["flag_plot"]:
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
            for i in np.arange(0, num_robot_time_steps, config["fin_every_n"]):
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
            input("Press Enter to continue...")
        return True

