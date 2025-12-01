import sys
import os

SCRIPT_PATH = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(SCRIPT_PATH, "../../../../"))

from typing import Optional, Callable
import numpy as np
import random
import scipy.interpolate as si
import scipy.spatial.transform as st
from diffusion_policy.common.replay_buffer import ReplayBuffer

from PyriteUtility.data_pipeline.indexing import (
    get_sample_ids,
    get_samples,
    get_dense_query_points_in_horizon,
)
from PyriteUtility.data_pipeline.data_plotting import plot_sample

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def get_val_mask(n_episodes, val_ratio, seed=0):
    val_mask = np.zeros(n_episodes, dtype=bool)
    if val_ratio <= 0:
        return val_mask

    # have at least 1 episode for validation, and at least 1 episode for train
    n_val = min(max(1, round(n_episodes * val_ratio)), n_episodes - 1)
    rng = np.random.default_rng(seed=seed)
    val_idxs = rng.choice(n_episodes, size=n_val, replace=False)
    val_mask[val_idxs] = True
    return val_mask


class StateBasedSequenceSampler:
    """
        No rgb or depth. Data are counted by robot states instead of rgb frames.
    """

    def __init__(
        self,
        shape_meta: dict,
        replay_buffer: dict,
        obs_to_obs_sample: Callable,
        action_to_action_sample: Callable,
        id_list: list,
        sparse_query_frequency_down_sample_steps: int = 1,
        episode_mask: Optional[np.ndarray] = None,
        action_padding: bool = False,
    ):
        self.flag_has_gripper = (
            "robot0_gripper" in shape_meta["sample"]["obs"]["sparse"]
        )
        episode_keys = replay_buffer["data"].keys()
        # Step one: Find the usable length of each episode
        episodes_length = replay_buffer["meta"]["episode_robot0_len"][:]
        episodes_length_for_query = episodes_length.copy()
        episodes_start = episodes_length.copy()

        buffer_time_ms = shape_meta["sample"][
            "training_duration_per_sparse_query"
        ]
        episode_count = -1
        for episode in episode_keys:
            episode_count += 1

            ##
            ## Find feasible start and end times for this episode
            ##
            end_time = np.inf
            start_time = -np.inf
            for id in id_list:
                ## I. start time: find a time after all the obs horizons
                ## I.2 robot
                if f"robot{id}_eef_pos" in shape_meta["sample"]["obs"]["sparse"]:
                    robot_start_id = (
                        shape_meta["sample"]["obs"]["sparse"][f"robot{id}_eef_pos"][
                            "horizon"
                        ]
                        * shape_meta["sample"]["obs"]["sparse"][f"robot{id}_eef_pos"][
                            "down_sample_steps"
                        ]
                    ) + 1
                else:
                    assert f"robot{id}_js" in shape_meta["sample"]["obs"]["sparse"]
                    robot_start_id = (
                        shape_meta["sample"]["obs"]["sparse"][f"robot{id}_js"][
                            "horizon"
                        ]
                        * shape_meta["sample"]["obs"]["sparse"][f"robot{id}_js"][
                            "down_sample_steps"
                        ]
                    ) + 1

                robot_start_time = np.squeeze(
                    replay_buffer["data"][episode]["obs"][
                        f"robot_time_stamps_{id}"
                    ][robot_start_id]
                )

                ## I.3 wrench (optional)
                if f"robot{id}_eef_wrench" in shape_meta["sample"]["obs"]["sparse"]:
                    wrench_start_id = (
                        shape_meta["sample"]["obs"]["sparse"][f"robot{id}_eef_wrench"][
                            "horizon"
                        ]
                        * shape_meta["sample"]["obs"]["sparse"][
                            f"robot{id}_eef_wrench"
                        ]["down_sample_steps"]
                    ) + 1

                    wrench_start_time = np.squeeze(
                        replay_buffer["data"][episode]["obs"][
                            f"wrench_time_stamps_{id}"
                        ][wrench_start_id]
                    )
                else:
                    wrench_start_time = -1e9

                ## I.4 gripper
                gripper_start_time = -1e9
                if self.flag_has_gripper:
                    gripper_start_id = (
                        shape_meta["sample"]["obs"]["sparse"][f"robot{id}_gripper"][
                            "horizon"
                        ]
                        * shape_meta["sample"]["obs"]["sparse"][
                            f"robot{id}_gripper"
                        ]["down_sample_steps"]
                    ) + 1
                    gripper_start_time = np.squeeze(
                        replay_buffer["data"][episode]["obs"][
                            f"gripper_time_stamps_{id}"
                        ][gripper_start_id]
                    )

                ## I.4 find max
                start_time = max(
                    start_time,
                    robot_start_time,
                    wrench_start_time,
                    gripper_start_time,
                )

                ## II. end time: the last time stamp before buffer_time_ms
                ## II.2 robot
                robot_end_time = np.squeeze(
                    replay_buffer["data"][episode]["obs"][
                        f"robot_time_stamps_{id}"
                    ][-1]
                )

                ## II.3 find min
                if not action_padding:
                    # if no action padding, truncate the indices to query so the last query point
                    #  still has access to the whole horizon of actions
                    #  This is enforced by limiting sparse action queries alone.
                    #  It is assumed that the dense action is not affected.
                    end_time = min(end_time, robot_end_time) - buffer_time_ms
                else:
                    end_time = min(end_time, robot_end_time)

                assert end_time > 0
                assert end_time > start_time

            last_robot_idx = 1e9
            first_robot_idx = -1
            for id in id_list:
                robot_times = np.squeeze(
                    replay_buffer["data"][episode]["obs"][f"robot_time_stamps_{id}"]
                )
                # find the last robot_times index that is before low_dim_end_time
                robot_id = np.searchsorted(robot_times, end_time, side="right") - 1
                last_robot_idx = min(last_robot_idx, robot_id)

                # find the first robot_times index that is after low_dim_start_time
                robot_id = np.searchsorted(robot_times, start_time, side="left")
                first_robot_idx = max(first_robot_idx, robot_id)

            episodes_length_for_query[episode_count] = last_robot_idx - first_robot_idx
            episodes_start[episode_count] = first_robot_idx
        assert np.min(episodes_length_for_query) > 0

        # Step two: Computes indices from episodes_length_for_query. indices[i] = (epi_id, epi_len, id)
        #   epi_id: which episode the index i belongs to.
        #   id: the index within the episode.
        epi_id = []
        ids = []
        episode_count = -1
        for key in episode_keys:
            episode_count += 1
            episode_index = int(key.split("_")[-1])
            if episode_mask is not None and not episode_mask[episode_count]:
                # skip episode
                continue

            # normal processing
            array_length = episodes_length_for_query[episode_count]
            ep_id_to_be_added = [episode_index] * array_length
            ids_to_be_added = episodes_start[episode_count] + np.arange(
                array_length
            )
            # Down sample the query indices to make the dataset smaller
            epi_id.extend(
                ep_id_to_be_added[::sparse_query_frequency_down_sample_steps]
            )
            ids.extend(ids_to_be_added[::sparse_query_frequency_down_sample_steps])

        indices = list(zip(epi_id, ids))
        print(f"Total usable episodes: {episode_count+1}, usable indices: {len(indices)}")
        # input("Press Enter to continue...")

        self.shape_meta = shape_meta
        self.replay_buffer = replay_buffer
        self.action_padding = action_padding
        self.indices = indices
        self.obs_to_obs_sample = obs_to_obs_sample
        self.action_to_action_sample = action_to_action_sample
        self.id_list = id_list

        self.ignore_rgb_is_applied = (
            False  # speed up the interation when getting normalizer
        )
        
    def __len__(self):
        return len(self.indices)

    def sample_sequence(self, idx):
        """Sample a sequence of observations and actions at idx."""
        epi_id, robot_id = self.indices[idx]
        episode = f"episode_{epi_id}"
        data_episode = self.replay_buffer["data"][episode]

        query_time = np.squeeze(data_episode["obs"]["robot_time_stamps_0"][robot_id])
        # query time is given for the robot0 obs data.
        # To get others (low dim, action), we need to find their id
        sparse_obs_unprocessed = dict()
        sparse_action_unprocessed = []
        for key, attr in self.shape_meta["sample"]["obs"]["sparse"].items():
            input_arr = data_episode["obs"][key]
            this_horizon = attr["horizon"]
            this_downsample_steps = attr["down_sample_steps"]
            type = self.shape_meta["obs"][key]["type"]

            if 'time' in key:
                id = int(key.split("_")[-1])
            elif "item" in key:
                id = int(key[10]) # item_poses0_xxxx
            else:
                id = int(key[5])  # robot0_xxxx

            if "wrench" in key: # robot0_eef_wrench
                time_stamp_key = f"wrench_time_stamps_{id}"
            elif "item" in key:
                time_stamp_key = f"item_time_stamps_{id}"
            else:
                time_stamp_key = f"robot_time_stamps_{id}"

            # find the query id for the query time
            query_id = np.searchsorted(
                np.squeeze(data_episode["obs"][time_stamp_key]), query_time
            )
            found_time = data_episode["obs"][time_stamp_key][query_id]

            if abs(found_time - query_time) > 50.0 and "policy" not in key:
                print("processing sparse key: ", key)
                print("sparse query_time: ", query_time)
                print(
                    "total time: ",
                    data_episode["obs"][time_stamp_key][-1],
                )
                print("query_id: ", query_id)
                print(
                    "total id: ",
                    len(data_episode["obs"][time_stamp_key]),
                )
                print(bcolors.WARNING + f"[sampler] {episode} Warning: closest data point at {found_time} is not equal to query_time {query_time}" + bcolors.ENDC)
                print("This is for key ", key)

            # how many obs frames before the query time are valid
            num_valid = min(this_horizon, query_id // this_downsample_steps + 1)
            slice_start = query_id - (num_valid - 1) * this_downsample_steps
            assert slice_start >= 0

            # sample every this_downsample_steps frames from slice_start to query_id+1,
            # then fill the rest with the first frame if needed
            if type == "rgb":
                if self.ignore_rgb_is_applied:
                    continue
                output = input_arr[slice_start : query_id + 1 : this_downsample_steps]
            elif type == "low_dim":
                output = input_arr[
                    slice_start : query_id + 1 : this_downsample_steps
                ].astype(np.float32)
                assert output.shape[0] == num_valid
            elif type == "timestamp":
                output = input_arr[
                    slice_start : query_id + 1 : this_downsample_steps
                ].astype(np.float32)
                assert output.shape[0] == num_valid
            # solve padding
            if output.shape[0] < this_horizon:
                padding = np.repeat(output[:1], this_horizon - output.shape[0], axis=0)
                output = np.concatenate([padding, output], axis=0)
            sparse_obs_unprocessed[key] = output

        # sparse action
        action_id = np.searchsorted(
            np.squeeze(data_episode["action_time_stamps"]), query_time
        )
        found_time = data_episode["action_time_stamps"][action_id]
        if abs(found_time - query_time) > 20.0:
            print(bcolors.WARNING + f"[sampler] {episode} Warning: action found_time {found_time} is not equal to query_time {query_time}" + bcolors.ENDC)

        if "sparse" in self.shape_meta["sample"]["action"]:
            input_arr = data_episode["action"]
            action_horizon = self.shape_meta["sample"]["action"]["sparse"]["horizon"]
            action_down_sample_steps = self.shape_meta["sample"]["action"]["sparse"][
                "down_sample_steps"
            ]
            slice_end = min(
                len(input_arr) - 1,
                action_id + (action_horizon - 1) * action_down_sample_steps + 1,
            )
            sparse_action_unprocessed = input_arr[
                action_id:slice_end:action_down_sample_steps
            ].astype(np.float32)
            # solve padding
            if not self.action_padding:
                if sparse_action_unprocessed.shape[0] != action_horizon:
                    print(sparse_action_unprocessed.shape[0], action_horizon)
                    print(action_id, slice_end, action_horizon, action_down_sample_steps)
                    print("Not enough points for action in episode ", episode, ", available points: ",  sparse_action_unprocessed.shape[0], ", required points: ", action_horizon)
                    # sparse_action_unprocessed = sparse_action_unprocessed[
                    #     :action_horizon
                    # ]
                assert sparse_action_unprocessed.shape[0] == action_horizon
            elif sparse_action_unprocessed.shape[0] < action_horizon:
                padding = np.repeat(
                    sparse_action_unprocessed[-1:],
                    action_horizon - sparse_action_unprocessed.shape[0],
                    axis=0,
                )
                sparse_action_unprocessed = np.concatenate(
                    [sparse_action_unprocessed, padding], axis=0
                )

        #   convert to relative pose
        obs_sample, base_pose = self.obs_to_obs_sample(
            obs_sparse=sparse_obs_unprocessed,
            obs_dense={},
            shape_meta=self.shape_meta,
            reshape_mode="reshape",
            id_list=self.id_list,
            ignore_rgb=self.ignore_rgb_is_applied,
        )
        action_sample = self.action_to_action_sample(
            action_sparse=sparse_action_unprocessed,
            action_dense={},
            id_list=self.id_list,
            base_pose=base_pose,
            shape_meta=self.shape_meta,
        )

        return obs_sample, action_sample

    def ignore_rgb(self, apply=True):
        self.ignore_rgb_is_applied = apply