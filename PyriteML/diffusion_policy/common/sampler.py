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


class SequenceSampler:
    """Sample sequences of observations and actions from replay buffer.
    Query keys are determined by the shape_meta.
    Query frequency is based on rgb data, which is likely to be the most sparse data.
    Other data corresponding to the query ID is obtain based on timestamps.
    1. Given query id, find the corresponding low dim/rgb id.
    1. Construct sparse sample:
        Sample sparse obs horizon before idx,
        Sample sparse action horizon after idx.
    2. Construct dense sample:
        Find the indices of dense query points. For each dense query point:
            Sample dense obs horizon before idx,
            Sample dense action horizon after idx.
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
        weighted_sampling: int = 1,     # if > 1, duplicate the correction sample's ids weighted_sampling times
        correction_horizon: int = 1,   # the times of action horizon to be counted as "start of correction"
        detect_correction_with_wrench: bool = False,  # if True, use wrench to detect correction
        new_episode_prob: float = 0.0,  # probability of sampling from new episodes
        num_new_episodes: int = 0,    # number of new episodes to consider
        correction_force_threshold: float = 2.0,  
        correction_torque_threshold: float = 1.0,
        num_initial_episodes: int = 10000,   # number of initial episodes to consider (the first num_initial_episodes episodes)
        use_raw_policy_timestamps: bool = False,
    ):
        self.flag_has_dense = "dense" in shape_meta["sample"]["obs"]
        self.flag_has_gripper = (
            "robot0_gripper" in shape_meta["sample"]["obs"]["sparse"]
        )
        episode_keys = replay_buffer["data"].keys()
        # Step one: Find the usable length of each episode
        episodes_length = replay_buffer["meta"]["episode_rgb0_len"][:]
        episodes_length_for_query = episodes_length.copy()
        episodes_start = episodes_length.copy()

        if self.flag_has_dense:
            buffer_time_ms = (
                shape_meta["sample"]["training_duration_per_sparse_query"]
                + shape_meta["sample"]["dense_action_duration_buffer_ms"]
            )
        else:
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

                ## I.1 RGB
                rgb_start_id = (
                    shape_meta["sample"]["obs"]["sparse"][f"rgb_{id}"]["horizon"]
                    * shape_meta["sample"]["obs"]["sparse"][f"rgb_{id}"][
                        "down_sample_steps"
                    ]
                ) + 1

                rgb_start_time = np.squeeze(
                    replay_buffer["data"][episode]["obs"][f"rgb_time_stamps_{id}"][
                        rgb_start_id
                    ]
                )

                ## I.2 robot
                robot_start_id = (
                    shape_meta["sample"]["obs"]["sparse"][f"robot{id}_eef_pos"][
                        "horizon"
                    ]
                    * shape_meta["sample"]["obs"]["sparse"][f"robot{id}_eef_pos"][
                        "down_sample_steps"
                    ]
                ) + 1

                if self.flag_has_dense:
                    dense_robot_start_id = (
                        shape_meta["sample"]["obs"]["dense"][f"robot{id}_eef_pos"][
                            "horizon"
                        ]
                        * shape_meta["sample"]["obs"]["dense"][
                            f"robot{id}_eef_pos"
                        ]["down_sample_steps"]
                    ) + 1
                    robot_start_id = max(robot_start_id, dense_robot_start_id)

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

                    if self.flag_has_dense:
                        dense_wrench_start_id = (
                            shape_meta["sample"]["obs"]["dense"][
                                f"robot{id}_eef_wrench"
                            ]["horizon"]
                            * shape_meta["sample"]["obs"]["dense"][
                                f"robot{id}_eef_wrench"
                            ]["down_sample_steps"]
                        ) + 1
                        wrench_start_id = max(wrench_start_id, dense_wrench_start_id)

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
                    rgb_start_time,
                    robot_start_time,
                    wrench_start_time,
                    gripper_start_time,
                )

                ## II. end time: the last time stamp before buffer_time_ms
                ## II.1 RGB
                rgb_end_time = np.squeeze(
                    replay_buffer["data"][episode]["obs"][f"rgb_time_stamps_{id}"][
                        -1
                    ]
                )

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
                    end_time = min(end_time, rgb_end_time, robot_end_time) - buffer_time_ms
                else:
                    end_time = min(end_time, rgb_end_time, robot_end_time)

                assert end_time > 0
                assert end_time > start_time

            last_rgb_idx = 1e9
            first_rgb_idx = -1
            for id in id_list:
                rgb_times = np.squeeze(
                    replay_buffer["data"][episode]["obs"][f"rgb_time_stamps_{id}"]
                )
                # find the last rgb_times index that is before low_dim_end_time
                rgb_id = np.searchsorted(rgb_times, end_time, side="right") - 1
                last_rgb_idx = min(last_rgb_idx, rgb_id)

                # find the first rgb_times index that is after low_dim_start_time
                rgb_id = np.searchsorted(rgb_times, start_time, side="left")
                first_rgb_idx = max(first_rgb_idx, rgb_id)

            episodes_length_for_query[episode_count] = last_rgb_idx - first_rgb_idx
            episodes_start[episode_count] = first_rgb_idx
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
            only_correction = episode_count >= num_initial_episodes
            print(f"only_correction: {only_correction}")
            if episode_mask is not None and not episode_mask[episode_count]:
                # skip episode
                continue

            has_mask = False
            if "mask_0" in replay_buffer["data"][key]["obs"]:
                masks = replay_buffer["data"][key]["obs"]["mask_0"]
                if np.any(masks):
                    has_mask = True

            if not has_mask:
                # normal processing
                array_length = episodes_length_for_query[episode_count]
                ep_id_to_be_added = [episode_index] * array_length
                ids_to_be_added = episodes_start[episode_count] + np.arange(
                    array_length
                )
                if not only_correction:
                    # Down sample the query indices to make the dataset smaller
                    epi_id.extend(
                        ep_id_to_be_added[::sparse_query_frequency_down_sample_steps]
                    )
                    ids.extend(ids_to_be_added[::sparse_query_frequency_down_sample_steps])

                if weighted_sampling > 1 or only_correction:
                    if "key_event_0" in replay_buffer["data"][key]["obs"]:
                        # use key event to detect correction
                        key_events = np.array(replay_buffer["data"][key]["obs"]["key_event_0"])
                        key_event_timestamps = np.array(replay_buffer["data"][key]["obs"]["key_event_time_stamps_0"])
                        rgb_timestamps = replay_buffer["data"][key]["obs"]["rgb_time_stamps_0"][ids_to_be_added[::sparse_query_frequency_down_sample_steps]]
                        # find the rgb timestamps that are closest to the key event timestamps == 1
                        correction_start_indices = np.where(key_events == 1)[0]
                        if len(correction_start_indices) > 0:
                            correction_start_timestamps = key_event_timestamps[correction_start_indices]
                            # find the closest rgb timestamps to the correction start timestamps
                            correction_start_ids = np.searchsorted(rgb_timestamps, correction_start_timestamps)
                            is_correction_sample = np.zeros(len(ids_to_be_added[::sparse_query_frequency_down_sample_steps]), dtype=bool)
                            is_correction_sample[correction_start_ids] = True

                            correction_end_indices = np.where(key_events == 0)[0]
                            correction_end_timestamps = key_event_timestamps[correction_end_indices]
                            flag_is_correction = np.zeros(len(ids_to_be_added[::sparse_query_frequency_down_sample_steps]), dtype=bool)
                            for start, end in zip(correction_start_timestamps, correction_end_timestamps):
                                start_id = np.searchsorted(rgb_timestamps, start)
                                end_id = np.searchsorted(rgb_timestamps, end, side="left")
                                flag_is_correction[start_id:end_id] = True
                            print("is correction samples:", np.sum(flag_is_correction), len(flag_is_correction))
                        else:
                            # no correction start timestamps found
                            is_correction_sample = np.zeros(len(ids_to_be_added[::sparse_query_frequency_down_sample_steps]), dtype=bool)
                            flag_is_correction = np.zeros(len(ids_to_be_added[::sparse_query_frequency_down_sample_steps]), dtype=bool)
                    else:
                        if detect_correction_with_wrench and "robot0_robot_wrench" in replay_buffer["data"][key]["obs"]:
                            # robot_wrench_0 = np.array(
                            #     replay_buffer["data"][key]["obs"]["robot0_robot_wrench"]
                            # )
                            wrench_0 = np.array(
                                replay_buffer["data"][key]["obs"]["robot0_eef_wrench"]
                            )
                            ATI_wrench_0 = wrench_0
                            wrench_horizon = shape_meta["sample"]["obs"]["sparse"]["robot0_eef_wrench"]["horizon"]
                            wrench_down_sample_steps = shape_meta["sample"]["obs"]["sparse"]["robot0_eef_wrench"]["down_sample_steps"]
                            sample_ids = np.searchsorted(
                                replay_buffer["data"][key]["obs"]["wrench_time_stamps_0"],
                                replay_buffer["data"][key]["obs"]["rgb_time_stamps_0"][
                                    ids_to_be_added[::sparse_query_frequency_down_sample_steps]
                                ],
                            ) - wrench_horizon*wrench_down_sample_steps
                            flag_not_in_correction = (np.linalg.norm(ATI_wrench_0[sample_ids, :3], axis=-1) < correction_force_threshold) & (np.linalg.norm(ATI_wrench_0[sample_ids, 3:], axis=-1) < correction_torque_threshold)
                            wrench_id_range = np.arange(0, wrench_horizon*wrench_down_sample_steps, wrench_down_sample_steps)
                            wrench_id_actual = np.clip(
                                sample_ids[:, None] + wrench_id_range, 0, len(ATI_wrench_0)-1
                            )
                            flag_before_correction = (np.mean(np.linalg.norm(ATI_wrench_0[wrench_id_actual, :3], axis=-1)) > correction_force_threshold) | (np.mean(np.linalg.norm(ATI_wrench_0[wrench_id_actual, 3:], axis=-1)) > correction_torque_threshold)
                            if only_correction:
                                flag_is_correction = (np.linalg.norm(ATI_wrench_0[sample_ids+wrench_horizon*wrench_down_sample_steps, :3], axis=-1) > correction_force_threshold) | (np.linalg.norm(ATI_wrench_0[sample_ids+wrench_horizon*wrench_down_sample_steps, 3:], axis=-1) > correction_torque_threshold)
                        else:
                            actions = np.array(replay_buffer["data"][key]["action"])

                            # detect correction occurance: if the first action is close to zero, and any of the other actions are not close to zero
                            reference_action = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
                            action_horizon = shape_meta["sample"]["action"]["sparse"]["horizon"]
                            action_down_sample_steps = shape_meta["sample"]["action"]["sparse"]["down_sample_steps"]
                            sample_ids = np.searchsorted(replay_buffer["data"][key]["action_time_stamps"], replay_buffer["data"][key]["obs"]["rgb_time_stamps_0"][ids_to_be_added[::sparse_query_frequency_down_sample_steps]])
                            
                            flag_not_in_correction = np.mean(np.abs(actions[sample_ids, :9] - reference_action[None, :]), axis=-1) < 1e-3

                            action_id_range = np.arange(1, action_horizon*action_down_sample_steps, action_down_sample_steps)
                            action_id_actual = np.clip(sample_ids[:, None] + action_id_range, 0, len(actions)-1)

                            flag_before_correction = np.mean(np.sum(np.abs(actions[action_id_actual, :9] - reference_action[None, None, :]), axis=1), axis=-1) > 1e-2
                            if only_correction:
                                flag_is_correction = np.mean(np.abs(actions[sample_ids, :9] - reference_action[None, :]), axis=-1) > 1e-2

                        is_correction_sample = flag_not_in_correction & flag_before_correction
                    correction_ids = ids_to_be_added[::sparse_query_frequency_down_sample_steps][is_correction_sample]

                    print(f"correction sample num {np.sum(is_correction_sample)}, total sample num {len(is_correction_sample)}")

                    # duplicate the correction sample's ids weighted_sampling times
                    for _ in range(weighted_sampling):
                        ids.extend(correction_ids)
                        epi_id.extend([episode_index] * np.sum(is_correction_sample))

                    if only_correction:
                        print(f"total correction sample num {np.sum(flag_is_correction)}")
                        epi_id.extend(
                            [episode_index] * np.sum(flag_is_correction)
                        )
                        ids.extend(ids_to_be_added[::sparse_query_frequency_down_sample_steps][flag_is_correction])

                    # sample more (correction horizon) before and after correction
                    if correction_horizon > 0:
                        addition_horizon = np.arange(sparse_query_frequency_down_sample_steps, (correction_horizon+1)*sparse_query_frequency_down_sample_steps, sparse_query_frequency_down_sample_steps)
                        additional_sample_ids = np.clip(correction_ids[:, None] + addition_horizon, ids_to_be_added[0], ids_to_be_added[-1])
                        additional_sample_ids = np.unique(additional_sample_ids.reshape(-1))
                        additional_sample_ids = additional_sample_ids[np.isin(additional_sample_ids, ids_to_be_added[::sparse_query_frequency_down_sample_steps][flag_is_correction])]
                        
                        additional_sample_ids = additional_sample_ids[~np.isin(additional_sample_ids, correction_ids)]     # filter out the additional sample ids that are already in the list
                        for _ in range(weighted_sampling):
                            ids.extend(additional_sample_ids)
                            epi_id.extend([episode_index] * len(additional_sample_ids))
                        print(f"additional sample ids num {len(additional_sample_ids)}")
            else:
                # now we have an episode with masks.
                # we need to find the end id of each continuous segment of ones in the mask array.
                # The processing of mask assumes that mask_0 and mask_1 are the same.
                mask_start_ids = np.array(np.where(np.diff(masks) == -1)) + 1
                robot_time_at_mask_start = replay_buffer["data"][key]["obs"][
                    f"robot_time_stamps_0"
                ][mask_start_ids]

                # find the corresponding rgb id
                rgb_id_at_mask_start = np.searchsorted(
                    np.squeeze(replay_buffer["data"][key]["obs"][f"rgb_time_stamps_0"]),
                    robot_time_at_mask_start,
                )

                array_length = len(rgb_id_at_mask_start)
                epi_id.extend([episode_index] * array_length)
                ids.extend(rgb_id_at_mask_start)

        indices = list(zip(epi_id, ids))

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

        self.new_episode_prob = new_episode_prob
        self.num_new_episodes = num_new_episodes    # number of new episodes to consider, latest episodes are considered
        # calculate sample weights for all the samples, old data has weight (1-new_episode_prob)/num_old, new data has weight new_episode_prob/num_new
        # new episodes are the latest num_new_episodes episodes
        self.sample_weights = np.ones(len(self.indices))
        if new_episode_prob > 0:
            # new episode: epi_id >= len(episode_keys) - num_new_episodes
            new_episode_mask = np.array([idx >= int(list(episode_keys)[-num_new_episodes].split("_")[-1]) for idx, _ in indices])
            if np.sum(new_episode_mask) == 0:
                print(bcolors.WARNING + f"[sampler] {episode} Warning: no new episodes found" + bcolors.ENDC)
            else:
                self.sample_weights[new_episode_mask] = new_episode_prob * (len(self.indices) - np.sum(new_episode_mask))
                self.sample_weights[~new_episode_mask] = (1 - new_episode_prob) * np.sum(new_episode_mask)
        self.use_raw_policy_timestamps = use_raw_policy_timestamps

    def __len__(self):
        return len(self.indices)

    def sample_sequence(self, idx):
        """Sample a sequence of observations and actions at idx."""
        epi_id, rgb_id = self.indices[idx]
        episode = f"episode_{epi_id}"
        data_episode = self.replay_buffer["data"][episode]

        query_time = np.squeeze(data_episode["obs"]["rgb_time_stamps_0"][rgb_id])
        # query time is given for the rgb0 obs data.
        # To get others (rgb, low dim, action), we need to find their id
        sparse_obs_unprocessed = dict()
        sparse_action_unprocessed = []
        for key, attr in self.shape_meta["sample"]["obs"]["sparse"].items():
            input_arr = data_episode["obs"][key]
            this_horizon = attr["horizon"]
            this_downsample_steps = attr["down_sample_steps"]
            type = self.shape_meta["obs"][key]["type"]

            if "rgb" in key or 'time' in key:
                id = int(key.split("_")[-1])
            elif "policy" in key:
                id = int(key[12])
            else:
                id = int(key[5])  # robot0_xxxx

            if "rgb" in key:
                time_stamp_key = f"rgb_time_stamps_{id}"
            elif "wrench" in key:
                time_stamp_key = f"wrench_time_stamps_{id}"
            elif "gripper" in key:
                time_stamp_key = f"gripper_time_stamps_{id}"
            elif "policy" in key:
                time_stamp_key = f"policy_time_stamps_{id}"
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
                if self.use_raw_policy_timestamps and "policy" in key:
                    policy_start_indices = np.arange(0, len(data_episode["obs"][time_stamp_key])+1, this_horizon)
                    policy_idx = np.searchsorted(policy_start_indices, query_id)
                    slice_start, slice_end = policy_start_indices[policy_idx-1], policy_start_indices[policy_idx]
                    output = input_arr[slice_start:slice_end]
                    assert output.shape[0] == this_horizon
                else:
                    output = input_arr[
                        slice_start : query_id + 1 : this_downsample_steps
                    ].astype(np.float32)
                    assert output.shape[0] == num_valid
            elif type == "timestamp":
                if self.use_raw_policy_timestamps and "policy" in key:
                    policy_start_indices = np.arange(0, len(data_episode["obs"][time_stamp_key])+1, this_horizon)
                    policy_idx = np.searchsorted(policy_start_indices, query_id)
                    slice_start, slice_end = policy_start_indices[policy_idx-1], policy_start_indices[policy_idx]
                    output = input_arr[slice_start:slice_end]
                    assert output.shape[0] == this_horizon
                else:
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

        # dense obs and action
        # Given one query time, we sample H groups of dense obs and action,
        # spanning across sparse_total_steps action time steps.
        dense_obs_unprocessed = {}
        dense_action_unprocessed = []

        if self.flag_has_dense:
            # 1. Compute dense query time points
            total_duration_ms = self.shape_meta["sample"][
                "training_duration_per_sparse_query"
            ]
            delta_ms = self.shape_meta["sample"][
                "training_delta_time_between_dense_queries_ms"
            ]
            dense_query_times = np.arange(
                query_time,
                query_time + total_duration_ms,
                delta_ms,
            )

            # dense obs
            for key, attr in self.shape_meta["sample"]["obs"]["dense"].items():
                this_horizon = attr["horizon"]
                this_downsample_steps = attr["down_sample_steps"]
                type = self.shape_meta["obs"][key]["type"]
                id = int(key[5])  # robot0_xxxx
                if "wrench" in key:
                    time_stamp_key = f"wrench_time_stamps_{id}"
                elif "gripper" in key:
                    time_stamp_key = f"gripper_time_stamps_{id}"
                else:
                    time_stamp_key = f"robot_time_stamps_{id}"

                # 2. find the query ids for the query times
                dense_query_ids = np.searchsorted(
                    np.squeeze(data_episode["obs"][time_stamp_key]), dense_query_times
                )
                found_times = data_episode["obs"][time_stamp_key][dense_query_ids]
                if any(abs(found_times - dense_query_times) > 10.0):
                    print("processing dense key: ", key)
                    print("dense_query_times: ", dense_query_times)
                    print("found_times: ", found_times)
                    print(
                        "total time: ",
                        data_episode["obs"][time_stamp_key][-1],
                    )
                    print("dense_query_ids: ", dense_query_ids)
                    print(
                        "total id: ",
                        len(data_episode["obs"][time_stamp_key]),
                    )
                    raise ValueError(
                        f"[sampler] {episode} Warning: some dense sample points are far from the found_times"
                    )

                # 3. get samples from id
                dense_obs_unprocessed[key] = get_samples(
                    data_episode["obs"][key],
                    dense_query_ids,
                    this_horizon,
                    this_downsample_steps,
                    backwards=True,
                    closed=True,
                ).astype(np.float32)

            # dense action (H, T, D)
            # assuming no padding is needed
            # 2. find the query id for the query time
            dense_action_query_ids = np.searchsorted(
                np.squeeze(data_episode["action_time_stamps"]), dense_query_times
            )
            # 3. get samples from id
            dense_action_horizon = self.shape_meta["sample"]["action"]["dense"][
                "horizon"
            ]
            dense_action_down_sample_steps = self.shape_meta["sample"]["action"][
                "dense"
            ]["down_sample_steps"]
            dense_action_unprocessed_ids = get_sample_ids(
                dense_action_query_ids,
                dense_action_horizon,
                dense_action_down_sample_steps,
                backwards=False,
                closed=True,
            )
            dense_action_unprocessed = data_episode["action"][
                dense_action_unprocessed_ids
            ].astype(
                np.float32
            )  # [H, T, D]

            # experimental: get corresponding action wrench
            dense_action_unprocessed_times = data_episode["action_time_stamps"][
                dense_action_unprocessed_ids
            ]  # [H, T]
            dense_action_wrench_list = []
            for id in self.id_list:
                dense_action_wrench_id = np.searchsorted(
                    np.squeeze(data_episode["obs"][f"wrench_time_stamps_{id}"]),
                    dense_action_unprocessed_times,
                )
                dense_action_wrench = data_episode["obs"][f"robot{id}_eef_wrench"][
                    dense_action_wrench_id
                ].astype(np.float32)
                dense_action_wrench_list.append(dense_action_wrench)
            dense_action_wrench = np.squeeze(
                np.stack(dense_action_wrench_list, axis=-1)
            )

        #   convert to relative pose
        obs_sample, base_pose = self.obs_to_obs_sample(
            obs_sparse=sparse_obs_unprocessed,
            obs_dense=dense_obs_unprocessed,
            shape_meta=self.shape_meta,
            reshape_mode="reshape",
            id_list=self.id_list,
            ignore_rgb=self.ignore_rgb_is_applied,
        )
        action_sample = self.action_to_action_sample(
            action_sparse=sparse_action_unprocessed,
            action_dense=dense_action_unprocessed,
            id_list=self.id_list,
            base_pose=base_pose,
            shape_meta=self.shape_meta,
        )

        if self.flag_has_dense:
            action_sample["dense_wrench"] = dense_action_wrench

        # # debug plotting
        # sparse_obs_down_sample_steps = self.shape_meta['sample']['obs']['sparse']['robot0_eef_pos']['down_sample_steps']
        # sparse_obs_horizon = self.shape_meta['sample']['obs']['sparse']['robot0_eef_pos']['horizon']
        # slice_start = low_dim_id - (sparse_obs_horizon - 1) * sparse_obs_down_sample_steps
        # sparse_obs_time_steps = np.arange(slice_start, low_dim_id + 1, sparse_obs_down_sample_steps).astype(np.float32)
        # sparse_action_time_steps = np.arange(action_id, slice_end, action_down_sample_steps)
        # total_time_steps = np.arange(slice_start, slice_end, 1)
        # total_obs = dict()
        # for key, attr in self.shape_meta['sample']['obs']['dense'].items():
        #     total_obs[key] = data_episode['obs'][key][total_time_steps]
        # total_action = data_episode['action'][total_time_steps]
        # plot_sample(total_obs, total_action, total_time_steps,
        #             # sparse_obs_unprocessed, dense_obs_unprocessed, sparse_action_unprocessed, dense_action_unprocessed,
        #             obs_sample['sparse'], obs_sample['dense'], action_sample['sparse'], action_sample['dense'],
        #             sparse_obs_time_steps,
        #             sparse_action_time_steps,
        #             dense_obs_queries, dense_obs_horizon, dense_obs_down_sample_steps,
        #             dense_action_queries, dense_action_horizon, dense_action_down_sample_steps
        # )
        # print('debug: printing')

        return obs_sample, action_sample

    def ignore_rgb(self, apply=True):
        self.ignore_rgb_is_applied = apply