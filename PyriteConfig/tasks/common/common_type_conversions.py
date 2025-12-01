import sys
import os

SCRIPT_PATH = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(SCRIPT_PATH, "../../../"))

import PyriteUtility.spatial_math.spatial_utilities as su
from PyriteUtility.computer_vision.computer_vision_utility import get_image_transform
from PyriteUtility.common import dict_apply

import numpy as np
from typing import Union, Dict, Optional
import zarr
import torch

##
## raw: keys used in the dataset. Each key contains data for a whole episode
## obs: keys used in inference. Needs some pre-processing before sending to the policy NN.
## obs_preprocessed: obs with normalized rgb keys. len = whole episode
## obs_sample: len = obs horizon, pose computed relative to current pose (id = -1)
## action: pose command in world frame. len = whole episode
## action_sample: len = action horizon, pose computed relative to current pose (id = 0)


def raw_to_obs(
    raw_data: Union[zarr.Group, Dict[str, np.ndarray]],
    episode_data: Union[zarr.Group, dict],
    shape_meta: dict,
    raw_policy_timestamp: bool = False,
):
    """convert shape_meta.raw data to shape_meta.obs.

    This function keeps image data as compressed zarr array in memory, while loads and decompresses
    low dim data.

    Args:
      raw_data: input, has keys from shape_meta.raw, each value is an ndarray of shape (t, ...)
      episode_data: output dictionary that matches shape_meta.obs
    """
    if isinstance(episode_data, zarr.Group):
        episode_data.create_group("obs")
    else:
        episode_data["obs"] = {}

    # obs.rgb: keep entry, keep as compressed zarr array in memory
    for key, attr in shape_meta["raw"].items():
        type = attr.get("type", "low_dim")
        if type == "rgb":
            # obs.rgb: keep as compressed zarr array in memory
            episode_data["obs"][key] = raw_data[key]

    # obs.low_dim: load entry, convert to obs.low_dim
    for id in shape_meta["id_list"]:
        if f"js_fb_{id}" in raw_data:
            js_fb = raw_data[f"js_fb_{id}"]
            episode_data["obs"][f"robot{id}_js"] = js_fb[:]
        
        if f"item_poses_{id}" in raw_data:
            item_poses = np.array(raw_data[f"item_poses_{id}"])
            n_items = item_poses.shape[1] // 7
            item_poses = item_poses.reshape(item_poses.shape[0], n_items, 7)
            item_poses9 = su.SE3_to_pose9(su.pose7_to_SE3(item_poses))
            episode_data["obs"][f"item_poses{id}_pos"] = item_poses9[..., :3].reshape(item_poses9.shape[0], -1)
            episode_data["obs"][f"item_poses{id}_rot_axis_angle"] = item_poses9[..., 3:].reshape(item_poses9.shape[0], -1)
            assert episode_data["obs"][f"item_poses{id}_pos"].shape[1] == n_items * 3
            assert episode_data["obs"][f"item_poses{id}_rot_axis_angle"].shape[1] == n_items * 6

        if f"ts_pose_fb_{id}" in raw_data:
            pose7_fb = raw_data[f"ts_pose_fb_{id}"]
            pose9_fb = su.SE3_to_pose9(su.pose7_to_SE3(pose7_fb))

            episode_data["obs"][f"robot{id}_eef_pos"] = pose9_fb[..., :3]
            episode_data["obs"][f"robot{id}_eef_rot_axis_angle"] = pose9_fb[..., 3:]

        if "robot0_eef_wrench" in shape_meta["obs"].keys():
            wrench = raw_data[f"wrench_{id}"]
            episode_data["obs"][f"robot{id}_eef_wrench"] = wrench[:]
            episode_data["obs"][f"wrench_time_stamps_{id}"] = raw_data[
                f"wrench_time_stamps_{id}"][:]
            
        if "policy_robot0_eef_pos" in shape_meta["obs"].keys():
            # residual policy with base action as input
            policy_pose7 = np.array(raw_data[f"policy_pose_command_{id}"])

            # find the closest policy command before or at the timestamp of the robot command
            policy_timestamps = raw_data[f"policy_time_stamps_{id}"][:]
            ts_timestamps = raw_data[f"robot_time_stamps_{id}"][:]

            policy_indices = np.searchsorted(policy_timestamps, ts_timestamps, side="right") - 1
            policy_indices = np.clip(policy_indices, 0, len(policy_pose7) - 1)
            if not raw_policy_timestamp:
                # align policy pose with robot
                policy_pose7 = policy_pose7[policy_indices]
                episode_data["obs"][f"policy_time_stamps_{id}"] = ts_timestamps
            else:
                # this is used for transformer residual policy
                episode_data["obs"][f"policy_time_stamps_{id}"] = policy_timestamps

            policy_pose9 = su.SE3_to_pose9(su.pose7_to_SE3(policy_pose7))
            episode_data["obs"][f"policy_robot{id}_eef_pos"] = policy_pose9[..., :3]
            episode_data["obs"][f"policy_robot{id}_eef_rot_axis_angle"] = policy_pose9[..., 3:]
        else:
            assert raw_policy_timestamp == False, "transformer without base action is Not yet implemented."

        if "policy_robot0_gripper" in shape_meta["obs"].keys():
            policy_gripper = raw_data[f"policy_gripper_command_{id}"][:]
            episode_data["obs"][f"policy_robot{id}_gripper"] = policy_gripper

        if "robot_wrench_0" in raw_data.keys():
            robot_wrench = raw_data[f"robot_wrench_{id}"]
            episode_data["obs"][f"robot{id}_robot_wrench"] = robot_wrench[:]

        if "key_event_0" in raw_data.keys():
            episode_data["obs"][f"key_event_{id}"] = raw_data[f"key_event_{id}"][:]
            episode_data["obs"][f"key_event_time_stamps_{id}"] = raw_data[f"key_event_time_stamps_{id}"][:]

        # optional: abs
        if "robot0_abs_eef_pos" in shape_meta["obs"].keys():
            episode_data["obs"][f"robot{id}_abs_eef_pos"] = pose9_fb[..., :3]
        if "robot0_abs_eef_rot_axis_angle" in shape_meta["obs"].keys():
            episode_data["obs"][f"robot{id}_abs_eef_rot_axis_angle"] = pose9_fb[..., 3:]

        # optional: gripper
        if "robot0_gripper" in shape_meta["obs"].keys():
            episode_data["obs"][f"robot{id}_gripper"] = raw_data[f"gripper_{id}"][:]
            episode_data["obs"][f"gripper_time_stamps_{id}"] = raw_data[
                f"gripper_time_stamps_{id}"
            ][:]

        # timestamps
        if f"robot_time_stamps_{id}" in raw_data:
            episode_data["obs"][f"robot_time_stamps_{id}"] = raw_data[
                f"robot_time_stamps_{id}"
            ][:]
        if f"rgb_time_stamps_{id}" in raw_data:
            episode_data["obs"][f"rgb_time_stamps_{id}"] = raw_data[
                f"rgb_time_stamps_{id}"
            ][:]
        if f"item_time_stamps_{id}" in raw_data:
            episode_data["obs"][f"item_time_stamps_{id}"] = raw_data[
                f"item_time_stamps_{id}"
            ][:]
        

def raw_to_action3(
    raw_data: Union[zarr.Group, Dict[str, np.ndarray]],
    episode_data: dict,
    id_list: list,
):
    """Convert shape_meta.raw data to shape_meta.action.
    Note: if relative action is used, the relative pose still needs to be computed every time a sample
    is made. This function only converts the whole episode, and does not know what pose to be relative to.

    >>> This function is currently a hack for residual policy testing. It assumes K=1500
    
    Args:
        raw_data: input, has keys from shape_meta.raw, each value is an ndarray of shape (t, ...)
        episode_data: output dictionary that has an 'action' field that matches shape_meta.action
    """
    action = []
    action_lens = []
    for id in id_list:
        # action: assemble from low_dim
        action_lens.append(raw_data[f"ts_pose_command_{id}"].shape[0])
        ts_pose7_command = raw_data[f"ts_pose_command_{id}"][:]
        ts_pose7_virtual_target = raw_data[f"ts_pose_virtual_target_{id}"][:]
        stiffness = raw_data[f"stiffness_{id}"][:][:, np.newaxis]
        delta_pos = ts_pose7_virtual_target[..., :3] - ts_pose7_command[..., :3]
        force = delta_pos * stiffness
        delta_pos_adjusted = force / 1500
        ts_pose7_vt_adjusted = ts_pose7_command
        ts_pose7_vt_adjusted[..., :3] = ts_pose7_command[..., :3] + delta_pos_adjusted
        
        if f"policy_pose_command_{id}" in raw_data.keys():
            SE3_vt_adjusted = su.pose7_to_SE3(ts_pose7_vt_adjusted)
            policy_SE3_command = su.pose7_to_SE3(raw_data[f"policy_pose_command_{id}"])
            # compute delta action using relative pose transformation in SE3 space
            delta_action = su.SE3_inv(policy_SE3_command) @ SE3_vt_adjusted
            ts_pose3_command = delta_action[..., :3, 3]
        else:
            ts_pose3_command = ts_pose7_vt_adjusted[..., :3]
        action.append(ts_pose3_command)
    # action: trim to the shortest length
    action_len = min(action_lens)
    action = [x[:action_len] for x in action]

    episode_data["action"] = np.concatenate(action, axis=-1)
    assert (
        episode_data["action"].shape[1] == 3 or episode_data["action"].shape[1] == 6
    )

    # action timestamps is set according to robot 0
    episode_data["action_time_stamps"] = raw_data["robot_time_stamps_0"][:action_len]


def raw_to_action7(
    raw_data: Union[zarr.Group, Dict[str, np.ndarray]],
    episode_data: dict,
    id_list: list,
):
    """Convert shape_meta.raw data to shape_meta.action.
    Note: if relative action is used, the relative pose still needs to be computed every time a sample
    is made. This function only converts the whole episode, and does not know what pose to be relative to.

    Args:
        raw_data: input, has keys from shape_meta.raw, each value is an ndarray of shape (t, ...)
        episode_data: output dictionary that has an 'action' field that matches shape_meta.action
    """
    action = []
    action_lens = []
    for id in id_list:
        # action: assemble from low_dim
        action_lens.append(raw_data[f"js_command_{id}"].shape[0])
        js_command = raw_data[f"js_command_{id}"][:]
        action.append(js_command)

    action_len = min(action_lens)
    action = [x[:action_len] for x in action]

    episode_data["action"] = np.concatenate(action, axis=-1)
    assert episode_data["action"].shape[1] == 7 or episode_data["action"].shape[1] == 14

    # action timestamps is set according to robot 0
    episode_data["action_time_stamps"] = raw_data["robot_time_stamps_0"][:action_len]


def raw_to_action9(
    raw_data: Union[zarr.Group, Dict[str, np.ndarray]],
    episode_data: dict,
    id_list: list,
    shape_meta: dict,
):
    """Convert shape_meta.raw data to shape_meta.action.
    Note: if relative action is used, the relative pose still needs to be computed every time a sample
    is made. This function only converts the whole episode, and does not know what pose to be relative to.

    Args:
        raw_data: input, has keys from shape_meta.raw, each value is an ndarray of shape (t, ...)
        episode_data: output dictionary that has an 'action' field that matches shape_meta.action
    """
    action = []
    action_lens = []
    for id in id_list:
        # action: assemble from low_dim
        action_lens.append(raw_data[f"ts_pose_command_{id}"].shape[0])
        ts_pose7_command = raw_data[f"ts_pose_command_{id}"][:]
        # if f"policy_pose_command_{id}" in raw_data.keys():
        if "policy_robot0_eef_pos" in shape_meta["obs"].keys():
            ts_SE3_command = su.pose7_to_SE3(ts_pose7_command)
            policy_SE3_command = su.pose7_to_SE3(raw_data[f"policy_pose_command_{id}"])
            # compute delta action using relative pose transformation in SE3 space
            delta_action = su.SE3_inv(policy_SE3_command) @ ts_SE3_command
            ts_pose9_command = su.SE3_to_pose9(delta_action)
        else:
            ts_pose9_command = su.SE3_to_pose9(su.pose7_to_SE3(ts_pose7_command))
        action.append(ts_pose9_command)

    action_len = min(action_lens)
    action = [x[:action_len] for x in action]

    episode_data["action"] = np.concatenate(action, axis=-1)
    assert episode_data["action"].shape[1] == 9 or episode_data["action"].shape[1] == 18

    # action timestamps is set according to robot 0
    episode_data["action_time_stamps"] = raw_data["robot_time_stamps_0"][:action_len]


def raw_to_action15(
    raw_data: Union[zarr.Group, Dict[str, np.ndarray]],
    episode_data: dict,
    id_list: list,
    shape_meta: dict,
):
    """Convert shape_meta.raw data to shape_meta.action.
    Note: if relative action is used, the relative pose still needs to be computed every time a sample
    is made. This function only converts the whole episode, and does not know what pose to be relative to.

    Args:
        raw_data: input, has keys from shape_meta.raw, each value is an ndarray of shape (t, ...)
        episode_data: output dictionary that has an 'action' field that matches shape_meta.action
    """
    action = []
    action_lens = []
    for id in id_list:
        # action: assemble from low_dim
        action_lens.append(raw_data[f"ts_pose_command_{id}"].shape[0])
        # ts_pose9_command
        ts_pose7_command = raw_data[f"ts_pose_command_{id}"][:]
        ts_timestamps = raw_data[f"robot_time_stamps_{id}"][:]
        if f"policy_pose_command_{id}" in shape_meta["raw"].keys():
            # residual!
            policy_pose7 = np.array(raw_data[f"policy_pose_command_{id}"])

            # find the closest policy command before or at the timestamp of the robot command
            policy_timestamps = raw_data[f"policy_time_stamps_{id}"][:]

            policy_indices = np.searchsorted(policy_timestamps, ts_timestamps, side="right") - 1
            policy_indices = np.clip(policy_indices, 0, len(policy_pose7) - 1)
            policy_pose7 = policy_pose7[policy_indices]

            ts_SE3_command = su.pose7_to_SE3(ts_pose7_command)
            policy_SE3_command = su.pose7_to_SE3(policy_pose7)
            # compute delta action using relative pose transformation in SE3 space
            delta_action = su.SE3_inv(policy_SE3_command) @ ts_SE3_command
            ts_pose9_command = su.SE3_to_pose9(delta_action)
        else:
            ts_pose9_command = su.SE3_to_pose9(su.pose7_to_SE3(ts_pose7_command))
        #  downsample moving average wrench
        ts_timestamps = raw_data[f"robot_time_stamps_{id}"][:]
        wrench = raw_data[f"wrench_moving_average_{id}"][:]
        wrench_timestamps = raw_data[f"wrench_time_stamps_{id}"][:]
        # Find closest matching timestamps
        indices = np.searchsorted(wrench_timestamps, ts_timestamps, side="left")
        indices = np.clip(indices, 0, len(wrench_timestamps) - 1)  # Ensure indices are valid
        downsampled_wrench = wrench[indices]
        action.append(
            np.concatenate(
                [ts_pose9_command, downsampled_wrench], axis=-1
            )
        )
    # action: trim to the shortest length
    action_len = min(action_lens)
    action = [x[:action_len] for x in action]

    episode_data["action"] = np.concatenate(action, axis=-1)
    assert (
        episode_data["action"].shape[1] == 15 or episode_data["action"].shape[1] == 30
    )

    # action timestamps is set according to robot 0
    episode_data["action_time_stamps"] = raw_data["robot_time_stamps_0"][:action_len]


def raw_to_action16(
    raw_data: Union[zarr.Group, Dict[str, np.ndarray]],
    episode_data: dict,
    id_list: list,
    shape_meta: dict,
):
    """Convert shape_meta.raw data to shape_meta.action.
    Note: if relative action is used, the relative pose still needs to be computed every time a sample
    is made. This function only converts the whole episode, and does not know what pose to be relative to.

    Args:
        raw_data: input, has keys from shape_meta.raw, each value is an ndarray of shape (t, ...)
        episode_data: output dictionary that has an 'action' field that matches shape_meta.action
    """
    action = []
    action_lens = []
    for id in id_list:
        # action: assemble from low_dim
        action_lens.append(raw_data[f"ts_pose_command_{id}"].shape[0])
        # ts_pose9_command
        ts_pose7_command = raw_data[f"ts_pose_command_{id}"][:]
        if f"policy_robot{id}_eef_pos" in shape_meta["obs"].keys():
            ts_SE3_command = su.pose7_to_SE3(ts_pose7_command)
            policy_SE3_command = su.pose7_to_SE3(raw_data[f"policy_pose_command_{id}"])
            # compute delta action using relative pose transformation in SE3 space
            delta_action = su.SE3_inv(policy_SE3_command) @ ts_SE3_command
            ts_pose9_command = su.SE3_to_pose9(delta_action)
        else:
            ts_pose9_command = su.SE3_to_pose9(su.pose7_to_SE3(ts_pose7_command))
        #  downsample moving average wrench
        ts_pose_timestamps = raw_data[f"robot_time_stamps_{id}"][:]
        wrench = raw_data[f"wrench_moving_average_{id}"][:]
        wrench_timestamps = raw_data[f"wrench_time_stamps_{id}"][:]

        # gripper command
        gripper_command = raw_data[f"gripper_{id}"][:]
        gripper_command_timestamps = raw_data[f"gripper_time_stamps_{id}"][:]
        # Find closest matching timestamps for wrench
        indices = np.searchsorted(wrench_timestamps, ts_pose_timestamps, side="left")
        indices = np.clip(indices, 0, len(wrench_timestamps) - 1)  # Ensure indices are valid
        downsampled_wrench = wrench[indices]
        # Find closest matching timestamps for gripper
        indices = np.searchsorted(gripper_command_timestamps, ts_pose_timestamps, side="left")
        indices = np.clip(indices, 0, len(gripper_command_timestamps) - 1)  # Ensure indices are valid
        downsampled_gripper_command = gripper_command[indices]
        
        action.append(
            np.concatenate(
                [ts_pose9_command, downsampled_wrench, downsampled_gripper_command], axis=-1
            )
        )
    # action: trim to the shortest length
    action_len = min(action_lens)
    action = [x[:action_len] for x in action]

    episode_data["action"] = np.concatenate(action, axis=-1)
    assert (
        episode_data["action"].shape[1] == 16 or episode_data["action"].shape[1] == 32
    )

    # action timestamps is set according to robot 0
    episode_data["action_time_stamps"] = raw_data["robot_time_stamps_0"][:action_len]



def raw_to_action19(
    raw_data: Union[zarr.Group, Dict[str, np.ndarray]],
    episode_data: dict,
    id_list: list,
):
    """Convert shape_meta.raw data to shape_meta.action.
    Note: if relative action is used, the relative pose still needs to be computed every time a sample
    is made. This function only converts the whole episode, and does not know what pose to be relative to.

    Args:
        raw_data: input, has keys from shape_meta.raw, each value is an ndarray of shape (t, ...)
        episode_data: output dictionary that has an 'action' field that matches shape_meta.action
    """
    action = []
    action_lens = []
    for id in id_list:
        # action: assemble from low_dim
        action_lens.append(raw_data[f"ts_pose_command_{id}"].shape[0])
        ts_pose7_command = raw_data[f"ts_pose_command_{id}"][:]
        ts_pose7_virtual_target = raw_data[f"ts_pose_virtual_target_{id}"][:]
        if f"policy_pose_command_{id}" in raw_data.keys():
            ts_SE3_command = su.pose7_to_SE3(ts_pose7_command)
            policy_SE3_command = su.pose7_to_SE3(raw_data[f"policy_pose_command_{id}"])
            # compute delta action using relative pose transformation in SE3 space
            delta_action = su.SE3_inv(policy_SE3_command) @ ts_SE3_command
            ts_pose9_command = su.SE3_to_pose9(delta_action)
            
            # compute virtual target relative to the actual command
            ts_SE3_virtual_target = su.pose7_to_SE3(ts_pose7_virtual_target)
            delta_virtual_target = su.SE3_inv(policy_SE3_command) @ ts_SE3_virtual_target
            ts_pose9_virtual_target = su.SE3_to_pose9(delta_virtual_target)
        else:
            ts_pose9_command = su.SE3_to_pose9(su.pose7_to_SE3(ts_pose7_command))
            ts_pose9_virtual_target = su.SE3_to_pose9(
                su.pose7_to_SE3(ts_pose7_virtual_target)
            )
        stiffness = raw_data[f"stiffness_{id}"][:][:, np.newaxis]
        action.append(
            np.concatenate(
                [ts_pose9_command, ts_pose9_virtual_target, stiffness], axis=-1
            )
        )
    # action: trim to the shortest length
    action_len = min(action_lens)
    action = [x[:action_len] for x in action]

    episode_data["action"] = np.concatenate(action, axis=-1)
    assert (
        episode_data["action"].shape[1] == 19 or episode_data["action"].shape[1] == 38
    )

    # action timestamps is set according to robot 0
    episode_data["action_time_stamps"] = raw_data["robot_time_stamps_0"][:action_len]


def raw_to_action21(
    raw_data: Union[zarr.Group, Dict[str, np.ndarray]],
    episode_data: dict,
    id_list: list,
):
    """Convert shape_meta.raw data to shape_meta.action.
    Note: if relative action is used, the relative pose still needs to be computed every time a sample
    is made. This function only converts the whole episode, and does not know what pose to be relative to.

    Args:
        raw_data: input, has keys from shape_meta.raw, each value is an ndarray of shape (t, ...)
        episode_data: output dictionary that has an 'action' field that matches shape_meta.action
    """
    action = []
    action_lens = []
    for id in id_list:
        # action: assemble from low_dim
        action_lens.append(raw_data[f"ts_pose_command_{id}"].shape[0])
        ts_pose7_command = raw_data[f"ts_pose_command_{id}"][:]
        ts_pose7_virtual_target = raw_data[f"ts_pose_virtual_target_{id}"][:]
        if f"policy_pose_command_{id}" in raw_data.keys():
            ts_SE3_command = su.pose7_to_SE3(ts_pose7_command)
            policy_SE3_command = su.pose7_to_SE3(raw_data[f"policy_pose_command_{id}"])
            # compute delta action using relative pose transformation in SE3 space
            delta_action = su.SE3_inv(policy_SE3_command) @ ts_SE3_command
            ts_pose9_command = su.SE3_to_pose9(delta_action)
            
            # compute virtual target relative to the actual command
            ts_SE3_virtual_target = su.pose7_to_SE3(ts_pose7_virtual_target)
            delta_virtual_target = su.SE3_inv(ts_SE3_command) @ ts_SE3_virtual_target
            ts_pose9_virtual_target = su.SE3_to_pose9(delta_virtual_target)
        else:
            ts_pose9_command = su.SE3_to_pose9(su.pose7_to_SE3(ts_pose7_command))
            ts_pose9_virtual_target = su.SE3_to_pose9(su.pose7_to_SE3(ts_pose7_virtual_target))
        stiffness = raw_data[f"stiffness_{id}"][:][:, np.newaxis]  # (T,) -> (T, 1)
        gripper = raw_data[f"gripper_{id}"][:]  # (T, 1)
        grasping_force = raw_data[f"wrench_{id}"][:, 2:3]  # (T, 1)
        # Downsample wrench data to match the action length based on timestamps
        robot_timestamps = raw_data[f"robot_time_stamps_{id}"][:].reshape(-1)
        wrench_timestamps = raw_data[f"wrench_time_stamps_{id}"][:].reshape(-1)
        # Find closest matching timestamps
        indices = np.searchsorted(wrench_timestamps, robot_timestamps, side="left")
        indices = np.clip(indices, 0, len(wrench_timestamps) - 1)  # Ensure indices are valid
        downsampled_grasping_force = grasping_force[indices]
        action.append(
            np.concatenate(
                [ts_pose9_command, ts_pose9_virtual_target, stiffness, gripper, downsampled_grasping_force], axis=-1
            )
        )
    # action: trim to the shortest length
    action_len = min(action_lens)
    action = [x[:action_len] for x in action]

    episode_data["action"] = np.concatenate(action, axis=-1)
    assert (
        episode_data["action"].shape[1] == 21 or episode_data["action"].shape[1] == 42
    )

    # action timestamps is set according to robot 0
    episode_data["action_time_stamps"] = raw_data["robot_time_stamps_0"][:action_len]


def obs_rgb_preprocess(
    obs: dict,
    obs_output: dict,
    reshape_mode: str,
    shape_meta: dict,
):
    """Pre-process the rgb data in the obs dictionary as inputs to policy network.

    This function does the following to the rgb keys in the obs dictionary:
    * Unpack/unzip it, if the rgb data is still stored as a compressed zarr array (not recommended)
    * Reshape the rgb image, or just check its shape.
    * Convert uint8 (0~255) to float32 (0~1)
    * Move its axes from THWC to TCHW.
    Since this function unpacks the whole key, it should only be used for online inference.
    If used in training, so the data length is the obs horizon instead of the whole episode len.

    Args:
        obs: dict with keys from shape_meta.obs
        obs_output: dict with the same keys but processed images
        reshape_mode: One of 'reshape', 'check', or 'none'.
        shape_meta: the shape_meta from task.yaml
    """
    obs_shape_meta = shape_meta["obs"]
    for key, attr in obs_shape_meta.items():
        type = attr.get("type", "low_dim")
        shape = attr.get("shape")
        if type == "rgb":
            this_imgs_in = obs[key]
            t, hi, wi, ci = this_imgs_in.shape
            co, ho, wo = shape
            assert ci == co
            out_imgs = this_imgs_in
            if (ho != hi) or (wo != wi):
                if reshape_mode == "reshape":
                    tf = get_image_transform(
                        input_res=(wi, hi), output_res=(wo, ho), bgr_to_rgb=False
                    )
                    out_imgs = np.stack([tf(x) for x in this_imgs_in])
                elif reshape_mode == "check":
                    print(
                        f"[obs_rgb_preprocess] shape check failed! Require {ho}x{wo}, get {hi}x{wi}"
                    )
                    assert False
            if this_imgs_in.dtype == np.uint8 or this_imgs_in.dtype == np.int32:
                out_imgs = out_imgs.astype(np.float32) / 255

            # THWC to TCHW
            obs_output[key] = np.moveaxis(out_imgs, -1, 1)


def sparse_obs_to_obs_sample(
    obs_sparse: dict,  # each key: (T, D)
    shape_meta: dict,
    reshape_mode: str,
    id_list: list,
    ignore_rgb: bool = False,
):
    """Prepare a sample of sparse obs as inputs to policy network.

    After packing an obs dictionary with keys according to shape_meta.sample.obs.sparse, with
    length corresponding to the correct horizons, pass it to this function to get it ready
    for the policy network.

    It does two things:
        1. RGB: unpack, reshape, normalize, turn into float
        2. low dim: convert pose to relative pose, turn into float

    Args:
        obs_sparse: dict with keys from shape_meta['sample']['obs']['sparse']
        shape_meta: the shape_meta from task.yaml
        reshape_mode: One of 'reshape', 'check', or 'none'.
        ignore_rgb: if True, skip the rgb keys. Used when computing normalizers.
    return:
        sparse_obs_processed: dict with keys from shape_meta['sample']['obs']['sparse']
        base_SE3: the initial pose used for relative pose calculation
    """
    sparse_obs_processed = {}
    assert len(obs_sparse) > 0
    if not ignore_rgb:
        obs_rgb_preprocess(obs_sparse, sparse_obs_processed, reshape_mode, shape_meta)

    # copy all low dim keys
    for key, attr in shape_meta["obs"].items():
        type = attr.get("type", "low_dim")
        if type == "low_dim":
            sparse_obs_processed[key] = obs_sparse[key].astype(
                np.float32
            )  # astype() makes a copy
        if type == "timestamp" and key in obs_sparse:
            sparse_obs_processed[key] = obs_sparse[key].astype(
                np.float32
            )

    if f"robot{id_list[0]}_eef_pos" in shape_meta["sample"]["obs"]["sparse"]:
        # generate relative pose
        base_SE3_WT = []
        for id in id_list:
            # convert pose to mat
            SE3_WT = su.pose9_to_SE3(
                np.concatenate(
                    [
                        sparse_obs_processed[f"robot{id}_eef_pos"],
                        sparse_obs_processed[f"robot{id}_eef_rot_axis_angle"],
                    ],
                    axis=-1,
                )
            )

            # solve relative obs
            base_SE3_WT.append(SE3_WT[-1])
            SE3_base_i = su.SE3_inv(base_SE3_WT[id]) @ SE3_WT

            pose9_relative = su.SE3_to_pose9(SE3_base_i)
            sparse_obs_processed[f"robot{id}_eef_pos"] = pose9_relative[..., :3]
            sparse_obs_processed[f"robot{id}_eef_rot_axis_angle"] = pose9_relative[..., 3:]

            if f"policy_robot{id}_eef_pos" in sparse_obs_processed.keys():
                # use the same base as the one used in robot{i}_eef_pos
                policy_SE3_WT = su.pose9_to_SE3(
                    np.concatenate(
                        [
                            sparse_obs_processed[f"policy_robot{id}_eef_pos"],
                            sparse_obs_processed[f"policy_robot{id}_eef_rot_axis_angle"],
                        ],
                        axis=-1,
                    )
                )
                SE3_base_i_policy = su.SE3_inv(base_SE3_WT[id]) @ policy_SE3_WT
                policy_pose9_relative = su.SE3_to_pose9(SE3_base_i_policy)
                sparse_obs_processed[f"policy_robot{id}_eef_pos"] = policy_pose9_relative[..., :3]
                sparse_obs_processed[f"policy_robot{id}_eef_rot_axis_angle"] = policy_pose9_relative[..., 3:]

            if f"policy_robot{id}_gripper" in sparse_obs_processed.keys():
                sparse_obs_processed[f"policy_robot{id}_gripper"] = obs_sparse[
                    f"policy_robot{id}_gripper"
                ].astype(np.float32)

            if f"robot{id}_eef_wrench" in sparse_obs_processed:
                # solve relative wrench
                # Note:
                #   The correct way to compute relative wrench requires
                #   using a different adjoint matrix for each time step of wrench.
                #   This can be expensive when the number of wrench samples is large.
                #   As an approximation, we use the adjoint matrix of the last pose.
                #   When the wrench is reported in tool frame, SE3_i_base is the identity matrix.
                SE3_i_base = su.SE3_inv(SE3_base_i)[-1]
                wrench = su.transpose(su.SE3_to_adj(SE3_i_base)) @ np.expand_dims(
                    obs_sparse[f"robot{id}_eef_wrench"], -1
                )
                sparse_obs_processed[f"robot{id}_eef_wrench"] = np.squeeze(wrench)

            # double check the shape
            for key, attr in shape_meta["sample"]["obs"]["sparse"].items():
                sparse_obs_horizon = attr["horizon"]
                if shape_meta["obs"][key]["type"] == "low_dim":
                    assert len(sparse_obs_processed[key].shape) == 2  # (T, D)
                    assert sparse_obs_processed[key].shape[0] == sparse_obs_horizon
                elif shape_meta["obs"][key]["type"] == "timestamp" and key in obs_sparse:
                    assert len(sparse_obs_processed[key].shape) == 1 # (T,)
                    assert sparse_obs_processed[key].shape[0] == sparse_obs_horizon
                else:
                    if not ignore_rgb:
                        assert len(sparse_obs_processed[key].shape) == 4  # (T, C, H, W)
                        assert sparse_obs_processed[key].shape[0] == sparse_obs_horizon

        return sparse_obs_processed, base_SE3_WT
    else:
        # joint space, uses absolute joints for now
        assert f"robot{id_list[0]}_js" in shape_meta["sample"]["obs"]["sparse"]
        base_joints = []
        for id in id_list:
            joints = sparse_obs_processed[f"robot{id}_js"]
            # save the last joints as the "base"
            base_joints.append(joints[-1])

            # double check the shape
            for key, attr in shape_meta["sample"]["obs"]["sparse"].items():
                sparse_obs_horizon = attr["horizon"]
                if shape_meta["obs"][key]["type"] == "low_dim":
                    assert len(sparse_obs_processed[key].shape) == 2  # (T, D)
                    assert sparse_obs_processed[key].shape[0] == sparse_obs_horizon
                elif shape_meta["obs"][key]["type"] == "timestamp" and key in obs_sparse:
                    assert len(sparse_obs_processed[key].shape) == 1 # (T,)
                    assert sparse_obs_processed[key].shape[0] == sparse_obs_horizon

        return sparse_obs_processed, base_joints
        

def dense_obs_to_obs_sample(
    obs_dense: dict,  # each key: (H, T, D) (training) or (T, D) (testing)
    shape_meta: dict,
    SE3_WBase: list,
    id_list: list,
):
    """Prepare a sample of obs as inputs to policy network.

    After packing an obs dictionary with keys according to shape_meta.sample.obs.dense, with
    length corresponding to the correct horizons, pass it to this function to get it ready
    for the policy network.

    Since dense obs only contains low dim data, it only does the low dim part:
        low dim: convert pose to relative pose about the initial pose of the SPARSE horizon

    Args:
        obs_dense: dict with keys from shape_meta['sample']['obs']['dense']
        shape_meta: the shape_meta from task.yaml
        SE3_WBase: a list of current pose SE3s, one per robot. The initial pose used for relative pose calculation
    """
    dense_obs_processed = {}
    for key in shape_meta["sample"]["obs"]["dense"].keys():
        dense_obs_processed[key] = obs_dense[key].astype(
            np.float32
        )  # astype() makes a copy
    # get the length of the first key in the dictionary obs_dense

    data_shape = next(iter(obs_dense.values())).shape
    assert len(data_shape) == 3
    H = data_shape[0]

    # convert each dense horizon to the same relative pose
    for id in id_list:
        for step in range(H):
            # generate relative pose. Everything is (T, D)
            # convert pose to mat
            SE3_WT = su.pose9_to_SE3(
                np.concatenate(
                    [
                        obs_dense[f"robot{id}_eef_pos"][step],
                        obs_dense[f"robot{id}_eef_rot_axis_angle"][step],
                    ],
                    axis=-1,
                )
            )

            # solve relative obs
            SE3_BaseT = np.linalg.inv(SE3_WBase[id]) @ SE3_WT

            pose9_relative = su.SE3_to_pose9(SE3_BaseT).astype(np.float32)
            dense_obs_processed[f"robot{id}_eef_pos"][step] = pose9_relative[..., :3]
            dense_obs_processed[f"robot{id}_eef_rot_axis_angle"][step] = pose9_relative[
                ..., 3:
            ]

            # solve relative wrench
            # Note:
            #   The correct way to compute relative wrench requires
            #   using a different adjoint matrix for each time step of wrench.
            #   This can be expensive when the number of wrench samples is large.
            #   As an approximation, we use the adjoint matrix of the last pose.
            #   When the wrench is reported in tool frame, SE3_i_base is the identity matrix.
            SE3_i_base = su.SE3_inv(SE3_BaseT[-1])

            wrench_0 = su.transpose(su.SE3_to_adj(SE3_i_base)) @ np.expand_dims(
                obs_dense[f"robot{id}_eef_wrench"][step], -1
            )
            dense_obs_processed[f"robot{id}_eef_wrench"][step] = np.squeeze(
                wrench_0
            ).astype(np.float32)

    # double check the shape
    for key in shape_meta["sample"]["obs"]["dense"].keys():
        assert dense_obs_processed[key].shape[0] == H
        assert len(dense_obs_processed[key].shape) == 3  # (H, T, D)

    return dense_obs_processed


def obs_to_obs_sample(
    obs_sparse: dict,  # each key: (T, D)
    obs_dense: dict,  # each key: (H, T, D)
    shape_meta: dict,
    reshape_mode: str,
    id_list: list,
    ignore_rgb: bool = False,
):
    """Prepare a sample of obs as inputs to policy network.

    After packing an obs dictionary with keys according to shape_meta.obs, with
    length corresponding to the correct horizons, pass it to this function to get it ready
    for the policy network.

    It does two things:
        1. RGB: unpack, reshape, normalize, turn into float
        2. low dim: convert pose to relative pose, turn into float
    For sparse obs, it does both. For dense obs, it only does the low dim part, and all poses are
    computed relative to the same current pose (id = 0).

    Args:
        obs_sparse: dict with keys from shape_meta['sample']['obs']['sparse']
        obs_dense: dict with keys from shape_meta['sample']['obs']['dense']
        shape_meta: the shape_meta from task.yaml
        reshape_mode: One of 'reshape', 'check', or 'none'.
        ignore_rgb: if True, skip the rgb keys. Used when computing normalizers.
    """
    obs_processed = {"sparse": {}, "dense": {}}
    obs_processed["sparse"], base_pose_mat = sparse_obs_to_obs_sample(
        obs_sparse, shape_meta, reshape_mode, id_list, ignore_rgb
    )
    if len(obs_dense) > 0:
        obs_processed["dense"] = dense_obs_to_obs_sample(
            obs_dense, shape_meta, base_pose_mat, id_list
        )

    return obs_processed, base_pose_mat

def action3_to_action_sample(
    action_sparse: np.ndarray,  # (T, D), D = 3 or 6
    action_dense: np.ndarray,  # (H, T, D) not used
    id_list: list,
    base_pose: list,
    shape_meta: dict,
):
    """Prepare a sample of actions as labels for the policy network.

    This function is used in training. It takes a sample of actions (len = action_horizon)
    and convert the poses in it to be relative to the current pose (id = 0).

    """
    action_processed = {"sparse": {}, "dense": {}}
    T, D = action_sparse.shape
    if len(id_list) == 1:
        assert D == 3
    else:
        assert D == 6

    def action3_preprocess(action: np.ndarray, SE3_WBase: np.ndarray):
        if "policy_robot0_eef_pos" in shape_meta["obs"].keys():
            return action
        
    if len(id_list) == 1:
        action_processed["sparse"] = action3_preprocess(action_sparse, base_pose[0])
    else:
        action_processed["sparse"] = np.concatenate(
            [
                action3_preprocess(action_sparse[:, :3], base_pose[0]),
                action3_preprocess(action_sparse[:, 3:6], base_pose[1]),
            ],
            axis=-1,
        )

    if len(action_dense) > 0:
        # not implemented properly
        raise NotImplementedError

    # double check the shape
    assert action_processed["sparse"].shape == (T, D)
    if len(action_dense) > 0:
        assert action_processed["dense"].shape == action_dense.shape

    return action_processed

def action7_to_action_sample(
    action_sparse: np.ndarray,  # (T, D), D = 7 or 14
    action_dense: np.ndarray,  # (H, T, D) not used
    id_list: list,
    base_pose: list,
    shape_meta: dict,
):
    """Prepare a sample of actions as labels for the policy network.

    This function is used in training. It takes a sample of actions (len = action_horizon)
    and convert the poses in it to be relative to the current pose (id = 0).

    """
    action_processed = {"sparse": {}, "dense": {}}
    T, D = action_sparse.shape
    if len(id_list) == 1:
        assert D == 7
    else:
        assert D == 14

    if len(id_list) == 1:
        action_processed["sparse"] = action_sparse - base_pose[0]
    else:
        raise NotImplementedError

    # double check the shape
    assert action_processed["sparse"].shape == (T, D)
    return action_processed

def action9_to_action_sample(
    action_sparse: np.ndarray,  # (T, D), D = 9
    action_dense: np.ndarray,  # (H, T, D), D = 9
    id_list: list,
    base_pose: list,
    shape_meta: dict,
):
    """Prepare a sample of actions as labels for the policy network.

    This function is used in training. It takes a sample of actions (len = action_horizon)
    and convert the poses in it to be relative to the current pose (id = 0).

    """
    action_processed = {"sparse": {}, "dense": {}}
    if len(action_sparse) > 0:
        T, D = action_sparse.shape
        if len(id_list) == 1:
            assert D == 9
        else:
            assert D == 18
    if len(action_dense) > 0:
        H, T, D = action_dense.shape
        if len(id_list) == 1:
            assert D == 9
        else:
            assert D == 18

    def action9_preprocess(action: np.ndarray, SE3_WBase: np.ndarray):
        if "policy_robot0_eef_pos" in shape_meta["obs"].keys():
            return action
        # generate relative pose
        # convert pose to mat
        pose9 = action
        SE3 = su.pose9_to_SE3(pose9)

        # solve relative obs
        SE3_WBase_inv = su.SE3_inv(SE3_WBase)
        SE3_relative = SE3_WBase_inv @ SE3

        pose9_relative = su.SE3_to_pose9(SE3_relative)

        return pose9_relative

    if len(action_sparse) > 0:
        if len(id_list) == 1:
            action_processed["sparse"] = action9_preprocess(action_sparse, base_pose[0])
        else:
            action_processed["sparse"] = np.concatenate(
                [
                    action9_preprocess(action_sparse[:, :9], base_pose[0]),
                    action9_preprocess(action_sparse[:, 9:18], base_pose[1]),
                ],
                axis=-1,
            )

    if len(action_dense) > 0:
        action_processed["dense"] = np.zeros_like(action_dense)
        H = action_dense.shape[0]
        for step in range(H):
            if len(id_list) == 1:
                # generate relative pose
                # convert pose to mat
                pose9 = action_dense[step]  # Tx9
                SE3 = su.pose9_to_SE3(pose9)  # Tx4x4

                # solve relative obs
                SE3_relative = su.SE3_inv(base_pose[0]) @ SE3
                pose9_relative = su.SE3_to_pose9(SE3_relative)
                action_processed["dense"][step] = pose9_relative
            else:
                # generate relative pose
                # convert pose to mat
                pose90 = action_dense[step][:, :9]  # Tx9
                pose91 = action_dense[step][:, 9:18]  # Tx9
                SE30 = su.pose9_to_SE3(pose90)
                SE31 = su.pose9_to_SE3(pose91)

                # solve relative obs
                SE3_relative0 = su.SE3_inv(base_pose[0]) @ SE30
                SE3_relative1 = su.SE3_inv(base_pose[1]) @ SE31
                pose9_relative0 = su.SE3_to_pose9(SE3_relative0)
                pose9_relative1 = su.SE3_to_pose9(SE3_relative1)
                action_processed["dense"][step] = np.concatenate(
                    [
                        pose9_relative0,
                        pose9_relative1,
                    ],
                    axis=-1,
                )

    # double check the shape
    if len(action_sparse) > 0:
        assert action_processed["sparse"].shape == (T, D)
    if len(action_dense) > 0:
        assert action_processed["dense"].shape == action_dense.shape

    return action_processed


def action15_to_action_sample(
    action_sparse: np.ndarray,  # (T, D), D = 19 or 38
    action_dense: np.ndarray,  # (H, T, D), D = 9
    id_list: list,
    base_pose: list,
    shape_meta: dict,
):
    """Prepare a sample of actions as labels for the policy network.

    This function is used in training. It takes a sample of actions (len = action_horizon)
    and convert the poses in it to be relative to the current pose (id = 0).

    """
    action_processed = {"sparse": {}}
    T, D = action_sparse.shape
    if len(id_list) == 1:
        assert D == 15
    else:
        assert D == 30

    def action15_preprocess(action: np.ndarray, SE3_WBase: np.ndarray):
        if "policy_pose_command_0" in shape_meta["raw"].keys():
            # residual, already relative to the base action
            return action
        
        # generate relative pose
        # convert pose to mat
        pose9 = action[:, 0:9]
        wrench6 = action[:, 9:15]
        SE3 = su.pose9_to_SE3(pose9)

        # solve relative obs
        SE3_WBase_inv = su.SE3_inv(SE3_WBase)
        SE3_relative = SE3_WBase_inv @ SE3
        pose9_relative = su.SE3_to_pose9(SE3_relative)

        return np.concatenate([pose9_relative, wrench6], axis=-1)

    if len(id_list) == 1:
        action_processed["sparse"] = action15_preprocess(action_sparse, base_pose[0])
    else:
        action_processed["sparse"] = np.concatenate(
            [
                action15_preprocess(action_sparse[:, :15], base_pose[0]),
                action15_preprocess(action_sparse[:, 15:30], base_pose[1]),
            ],
            axis=-1,
        )

    if len(action_dense) > 0:
        # not implemented properly
        raise NotImplementedError

    # double check the shape
    assert action_processed["sparse"].shape == (T, D)
    if len(action_dense) > 0:
        assert action_processed["dense"].shape == action_dense.shape

    return action_processed

def action16_to_action_sample(
    action_sparse: np.ndarray,  # (T, D), D = 16 or 32
    action_dense: np.ndarray,  # (H, T, D), D = 9
    id_list: list,
    base_pose: list,
    shape_meta: dict,
):
    """Prepare a sample of actions as labels for the policy network.

    This function is used in training. It takes a sample of actions (len = action_horizon)
    and convert the poses in it to be relative to the current pose (id = 0).

    """
    action_processed = {"sparse": {}}
    T, D = action_sparse.shape
    if len(id_list) == 1:
        assert D == 16
    else:
        assert D == 32

    def action16_preprocess(action: np.ndarray, SE3_WBase: np.ndarray):
        if "policy_robot0_eef_pos" in shape_meta["obs"].keys():
            # already relative to the base action
            return action
        
        # generate relative pose
        # convert pose to mat
        pose9 = action[:, 0:9]
        wrench6 = action[:, 9:15]
        g1 = action[:, 15:16]  # gripper
        SE3 = su.pose9_to_SE3(pose9)

        # solve relative obs
        SE3_WBase_inv = su.SE3_inv(SE3_WBase)
        SE3_relative = SE3_WBase_inv @ SE3
        pose9_relative = su.SE3_to_pose9(SE3_relative)

        return np.concatenate([pose9_relative, wrench6, g1], axis=-1)

    if len(id_list) == 1:
        action_processed["sparse"] = action16_preprocess(action_sparse, base_pose[0])
    else:
        action_processed["sparse"] = np.concatenate(
            [
                action16_preprocess(action_sparse[:, :16], base_pose[0]),
                action16_preprocess(action_sparse[:, 16:32], base_pose[1]),
            ],
            axis=-1,
        )

    if len(action_dense) > 0:
        # not implemented properly
        raise NotImplementedError

    # double check the shape
    assert action_processed["sparse"].shape == (T, D)
    if len(action_dense) > 0:
        assert action_processed["dense"].shape == action_dense.shape

    return action_processed

def action19_to_action_sample(
    action_sparse: np.ndarray,  # (T, D), D = 19 or 38
    action_dense: np.ndarray,  # (H, T, D) not used
    id_list: list,
    base_pose: list,
    shape_meta: dict,
):
    """Prepare a sample of actions as labels for the policy network.

    This function is used in training. It takes a sample of actions (len = action_horizon)
    and convert the poses in it to be relative to the current pose (id = 0).

    """
    action_processed = {"sparse": {}, "dense": {}}
    T, D = action_sparse.shape
    if len(id_list) == 1:
        assert D == 19
    else:
        assert D == 38

    def action19_preprocess(action: np.ndarray, SE3_WBase: np.ndarray):
        if "policy_robot0_eef_pos" in shape_meta["obs"].keys():
            return action
        # generate relative pose
        # convert pose to mat
        pose9 = action[:, 0:9]
        pose9_vt = action[:, 9:18]
        stiffness = action[:, 18:19]
        SE3 = su.pose9_to_SE3(pose9)
        SE3_vt = su.pose9_to_SE3(pose9_vt)

        # solve relative obs
        SE3_WBase_inv = su.SE3_inv(SE3_WBase)
        SE3_relative = SE3_WBase_inv @ SE3
        SE3_vt_relative = SE3_WBase_inv @ SE3_vt
        pose9_relative = su.SE3_to_pose9(SE3_relative)
        pose9_vt_relative = su.SE3_to_pose9(SE3_vt_relative)

        return np.concatenate([pose9_relative, pose9_vt_relative, stiffness], axis=-1)

    if len(id_list) == 1:
        action_processed["sparse"] = action19_preprocess(action_sparse, base_pose[0])
    else:
        action_processed["sparse"] = np.concatenate(
            [
                action19_preprocess(action_sparse[:, :19], base_pose[0]),
                action19_preprocess(action_sparse[:, 19:38], base_pose[1]),
            ],
            axis=-1,
        )

    if len(action_dense) > 0:
        # not implemented properly
        raise NotImplementedError

    # double check the shape
    assert action_processed["sparse"].shape == (T, D)
    if len(action_dense) > 0:
        assert action_processed["dense"].shape == action_dense.shape

    return action_processed


def action21_to_action_sample(
    action_sparse: np.ndarray,  # (T, D), D = 21 or 42
    action_dense: np.ndarray,  # (H, T, D) not used
    id_list: list,
    base_pose: list,
    shape_meta: dict,
):
    """Prepare a sample of actions as labels for the policy network.

    This function is used in training. It takes a sample of actions (len = action_horizon)
    and convert the poses in it to be relative to the current pose (id = 0).

    """
    action_processed = {"sparse": {}, "dense": {}}
    T, D = action_sparse.shape
    if len(id_list) == 1:
        assert D == 21
    else:
        assert D == 42

    def action21_preprocess(action: np.ndarray, SE3_WBase: np.ndarray):
        if "policy_robot0_eef_pos" in shape_meta["obs"].keys():
            return action
        # generate relative pose
        # convert pose to mat
        pose9 = action[:, 0:9]
        pose9_vt = action[:, 9:18]
        stiffness = action[:, 18:19]
        gripper = action[:, 19:20]
        grasping_force = action[:, 20:21]
        SE3 = su.pose9_to_SE3(pose9)
        SE3_vt = su.pose9_to_SE3(pose9_vt)

        # solve relative obs
        SE3_WBase_inv = su.SE3_inv(SE3_WBase)
        SE3_relative = SE3_WBase_inv @ SE3
        SE3_vt_relative = SE3_WBase_inv @ SE3_vt
        pose9_relative = su.SE3_to_pose9(SE3_relative)
        pose9_vt_relative = su.SE3_to_pose9(SE3_vt_relative)

        return np.concatenate(
            [pose9_relative, pose9_vt_relative, stiffness, gripper, grasping_force], axis=-1
        )

    if len(id_list) == 1:
        action_processed["sparse"] = action21_preprocess(action_sparse, base_pose[0])
    else:
        action_processed["sparse"] = np.concatenate(
            [
                action21_preprocess(action_sparse[:, :21], base_pose[0]),
                action21_preprocess(action_sparse[:, 21:42], base_pose[1]),
            ],
            axis=-1,
        )

    if len(action_dense) > 0:
        # not implemented properly
        raise NotImplementedError

    # double check the shape
    assert action_processed["sparse"].shape == (T, D)
    if len(action_dense) > 0:
        assert action_processed["dense"].shape == action_dense.shape

    return action_processed


def actionJS_postprocess(
    action: np.ndarray, current_joints: np.ndarray, delta_joint_limit=None
):
    """Convert policy outputs from delta joints to absolute joints
    Used in online inference
    
    args:
        action: (T, D), delta joints
        current_joints: (D,)
    """
    if delta_joint_limit is not None:
        action = np.clip(action, -delta_joint_limit, delta_joint_limit)

    action_absolute = current_joints + action

    # return pose matrices
    return action_absolute


def action9_postprocess(
    action: np.ndarray, current_SE3: list, id_list: list, fix_orientation=False, delta_pos_limit=None
):
    """Convert policy outputs from relative pose to world frame pose
    Used in online inference
    """

    action_SE3_absolute = [np.array] * len(id_list)
    for id in id_list:
        action_pose9 = action[..., 9 * id + 0 : 9 * id + 9]

        # TODO: apply limit here
        if delta_pos_limit is not None:
            delta_pos = action_pose9[:, :3]
            delta_pos = np.clip(delta_pos, -delta_pos_limit, delta_pos_limit)
            action_pose9[:, :3] = delta_pos

        action_SE3 = su.pose9_to_SE3(action_pose9)

        action_SE3_absolute[id] = current_SE3[id] @ action_SE3

        if fix_orientation:
            action_SE3_absolute[id][:, :3, :3] = current_SE3[id][:3, :3]

    # return pose matrices
    return action_SE3_absolute


def action10_postprocess(
    action: np.ndarray, current_SE3: list, id_list: list, fix_orientation=False
):
    """Convert policy outputs from relative pose to world frame pose
    Used in online inference
    """
    action_SE3_absolute = [np.array] * len(id_list)
    eoat = [np.array] * len(id_list)

    for id in id_list:
        action_pose9 = action[..., 10 * id + 0 : 10 * id + 9]
        eoat[id] = action[..., 10 * id + 9: 10 * id + 10]
        action_SE3 = su.pose9_to_SE3(action_pose9)

        action_SE3_absolute[id] = current_SE3[id] @ action_SE3

        if fix_orientation:
            action_SE3_absolute[id][:, :3, :3] = current_SE3[:3, :3]

    # return pose matrices
    return action_SE3_absolute, eoat

def action19_postprocess(
    action: np.ndarray, current_SE3: list, id_list: list, fix_orientation=False
):
    """Convert policy outputs from relative pose to world frame pose
    Used in online inference
    """

    action_SE3_absolute = [np.array] * len(id_list)
    action_SE3_vt_absolute = [np.array] * len(id_list)
    stiffness = [0] * len(id_list)

    for id in id_list:
        action_pose9 = action[..., 19 * id + 0 : 19 * id + 9]
        action_pose9_vt = action[..., 19 * id + 9 : 19 * id + 18]
        stiffness[id] = action[..., 19 * id + 18]
        action_SE3 = su.pose9_to_SE3(action_pose9)
        action_SE3_vt = su.pose9_to_SE3(action_pose9_vt)

        action_SE3_absolute[id] = current_SE3[id] @ action_SE3
        action_SE3_vt_absolute[id] = current_SE3[id] @ action_SE3_vt

        if fix_orientation:
            action_SE3_absolute[id][:, :3, :3] = current_SE3[:3, :3]
            action_SE3_vt_absolute[id][:, :3, :3] = current_SE3[:3, :3]

    # return pose matrices
    return action_SE3_absolute, action_SE3_vt_absolute, stiffness

def action21_postprocess(
    action: np.ndarray, current_SE3: list, id_list: list, fix_orientation=False
):
    """Convert policy outputs from relative pose to world frame pose
    Used in online inference
    """
    
    action_SE3_absolute = [np.array] * len(id_list)
    action_SE3_vt_absolute = [np.array] * len(id_list)
    stiffness = [0] * len(id_list)
    eoat = [np.array] * len(id_list)

    for id in id_list:
        action_pose9 = action[..., 21 * id + 0 : 21 * id + 9]
        action_pose9_vt = action[..., 21 * id + 9 : 21 * id + 18]
        stiffness[id] = action[..., 21 * id + 18]
        eoat[id] = action[..., 21 * id + 19: 21 * id + 21]
        action_SE3 = su.pose9_to_SE3(action_pose9)
        action_SE3_vt = su.pose9_to_SE3(action_pose9_vt)

        action_SE3_absolute[id] = current_SE3[id] @ action_SE3
        action_SE3_vt_absolute[id] = current_SE3[id] @ action_SE3_vt

        if fix_orientation:
            action_SE3_absolute[id][:, :3, :3] = current_SE3[:3, :3]
            action_SE3_vt_absolute[id][:, :3, :3] = current_SE3[:3, :3]

    # return pose matrices
    return action_SE3_absolute, action_SE3_vt_absolute, stiffness, eoat