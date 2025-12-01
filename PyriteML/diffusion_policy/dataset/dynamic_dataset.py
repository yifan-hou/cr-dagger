import sys
import os

SCRIPT_PATH = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(SCRIPT_PATH, "../../../"))

import copy
from typing import Dict, Optional

import numpy as np
import torch
import zarr
from threadpoolctl import threadpool_limits
from tqdm import trange, tqdm
from einops import rearrange, reduce

from diffusion_policy.common.normalize_util import (
    array_to_stats,
    concatenate_normalizer,
    get_identity_normalizer_from_stat,
    get_image_identity_normalizer,
    get_range_normalizer_from_stat,
)
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import SequenceSampler, get_val_mask
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.model.common.normalizer import LinearNormalizer

from PyriteConfig.tasks.common.common_type_conversions import (
    raw_to_obs,
    raw_to_action3,
    raw_to_action9,
    raw_to_action15,
    raw_to_action16,
    raw_to_action19,
    raw_to_action21,
    obs_to_obs_sample,
    action3_to_action_sample,
    action9_to_action_sample,
    action15_to_action_sample,
    action16_to_action_sample,
    action19_to_action_sample,
    action21_to_action_sample,
)

from PyriteUtility.data_pipeline.processing_functions import process_one_episode_into_zarr, generate_meta_for_zarr, compute_vt_for_episode
from PyriteUtility.computer_vision.imagecodecs_numcodecs import register_codecs

register_codecs()


class DynamicDataset(BaseImageDataset):
    def __init__(
        self,
        shape_meta: dict,
        sparse_query_frequency_down_sample_steps: int = 1,
        action_padding: bool = False,
        temporally_independent_normalization: bool = False,
        seed: int = 42,
        val_ratio: float = 0.0,
        normalize_wrench: bool = False,
        weighted_sampling: int = 1,
        correction_horizon: int = 1,
        virtual_target_config: dict = None,
        new_episode_prob: float = 0.0,
        correction_force_threshold: float = 3.0,
        correction_torque_threshold: float = 1.0,
        use_raw_policy_timestamps: bool = False,
    ):
        action_type = ""
        if (
            shape_meta["action"]["shape"][0] == 3
            or shape_meta["action"]["shape"][0] == 6
        ):
            action_type = "pose3"
        elif (
            shape_meta["action"]["shape"][0] == 9
            or shape_meta["action"]["shape"][0] == 18
        ):
            action_type = "pose9"
        elif (
            shape_meta["action"]["shape"][0] == 15
            or shape_meta["action"]["shape"][0] == 30
        ):
            action_type = "pose9wrench6"
        elif (
            shape_meta["action"]["shape"][0] == 16
            or shape_meta["action"]["shape"][0] == 32
        ):
            action_type = "pose9wrench6g1"
        elif (
            shape_meta["action"]["shape"][0] == 19
            or shape_meta["action"]["shape"][0] == 38
        ):
            action_type = "pose9pose9s1"
        elif (
            shape_meta["action"]["shape"][0] == 21
            or shape_meta["action"]["shape"][0] == 42
        ):
            action_type = "pose9pose9s1g2"
        else:
            raise RuntimeError("unsupported")
        self.action_type = action_type
        self.id_list = shape_meta["id_list"]

        if action_type == "pose3":
            action_to_action_sample = action3_to_action_sample
        elif action_type == "pose9":
            action_to_action_sample = action9_to_action_sample
        elif action_type == "pose9wrench6":
            action_to_action_sample = action15_to_action_sample
        elif action_type == "pose9wrench6g1":
            action_to_action_sample = action16_to_action_sample
        elif action_type == "pose9pose9s1":
            action_to_action_sample = action19_to_action_sample
        elif action_type == "pose9pose9s1g2":
            action_to_action_sample = action21_to_action_sample
        else:
            raise RuntimeError("unsupported")
        

        self.shape_meta = shape_meta
        self.val_ratio = val_ratio
        self.seed = seed
        self.action_type = action_type
        self.sparse_query_frequency_down_sample_steps = sparse_query_frequency_down_sample_steps
        self.action_to_action_sample = action_to_action_sample
        self.weighted_sampling = weighted_sampling
        self.correction_horizon = correction_horizon
        self.sparse_query_frequency_down_sample_steps = (
            sparse_query_frequency_down_sample_steps
        )
        self.action_padding = action_padding
        self.temporally_independent_normalization = temporally_independent_normalization
        self.threadpool_limits_is_applied = False
        self.normalize_wrench = normalize_wrench
        self.virtual_target_config = virtual_target_config

        self.new_episode_prob = new_episode_prob
        self.sampling_weights = None
        self.correction_force_threshold = correction_force_threshold
        self.correction_torque_threshold = correction_torque_threshold
        self.num_initial_episodes = None
        self.use_raw_policy_timestamps = use_raw_policy_timestamps

    ### Load the initial dataset from the processed zarr file
    #   Key member variables:
    #       ds_root: directory store, maintains the zarr file
    #       replay_buffer_raw: ReplayBuffer, used to load zarr into memory store
    #                          and check chunking.
    #                          Contains raw keys.
    #       replay_buffer_dict: a dict() that stores obs and action keys, pointing
    #                           to the corresponding memory in replay_buffer_raw.
    def load_initial_dataset(self, dataset_path: str, num_of_initial_episodes, num_of_new_episodes = 0):
        ## read existing data from file
        processed_dataset_path = dataset_path + "processed"
        print("[DynamicDataset] loading initial zarr data into store from", processed_dataset_path)
        # ‘r’ means read only (must exist); ‘r+’ means read/write (must exist); ‘a’ means read/write (create if doesn’t exist); ‘w’ means create (overwrite if exists); ‘w-’ means create (fail if exists).
        directory_store = zarr.DirectoryStore(path=processed_dataset_path)
        directory_root = zarr.open(store=directory_store, mode="a")
        
        if "data" in directory_root and "meta" in directory_root:
            print(f"[DynamicDataset] data and meta found in store with num of episodes = {len(directory_root['data'])}")
        else:
            print("[DynamicDataset] 'data' and 'meta' not found in store. Are you starting a new dataset?")
            input("[DynamicDataset] Press Enter to create empty directory store.")
            directory_root.create_group('data', overwrite=True)
            meta_group = directory_root.create_group('meta', overwrite=True)
            episode_rgb0_len = meta_group.create_dataset(
                "episode_rgb0_len",
                shape=(0,),
                dtype=np.int64,
                compressor=None,
                overwrite=False,
            )

        # load into memory store
        replay_buffer_raw = ReplayBuffer.copy_from_store(
            src_store=directory_store, dest_store=zarr.MemoryStore()
        )

        # convert raw to obs and action
        print("[DynamicDataset] raw to obs/action conversion")
        replay_buffer_dict = self.raw_episodes_conversion(replay_buffer_raw, self.shape_meta)

        # train/val mask for training
        val_mask = None
        train_mask = None
        sampler = None
        if replay_buffer_raw.n_episodes > 0:
            val_mask = get_val_mask(
                n_episodes=replay_buffer_raw.n_episodes, val_ratio=self.val_ratio, seed=self.seed
            )
            train_mask = ~val_mask
            print("[DynamicDataset] creating SequenceSampler.")
            sampler = SequenceSampler(
                shape_meta=self.shape_meta,
                replay_buffer=replay_buffer_dict,
                sparse_query_frequency_down_sample_steps=self.sparse_query_frequency_down_sample_steps,
                episode_mask=train_mask,
                action_padding=self.action_padding,
                obs_to_obs_sample=obs_to_obs_sample,
                action_to_action_sample=self.action_to_action_sample,
                id_list=self.id_list,
                weighted_sampling=self.weighted_sampling,
                correction_horizon=self.correction_horizon,
                detect_correction_with_wrench = True,
                correction_force_threshold=self.correction_force_threshold,
                correction_torque_threshold=self.correction_torque_threshold,
                num_initial_episodes=num_of_initial_episodes,
                num_new_episodes=num_of_new_episodes,
                use_raw_policy_timestamps=self.use_raw_policy_timestamps,
            )

        self.val_mask = val_mask
        self.sampler = sampler
        self.ds_root = directory_root
        self.replay_buffer_raw = replay_buffer_raw
        self.replay_buffer_dict = replay_buffer_dict
        self.dataset_path = dataset_path
        self.num_initial_episodes = num_of_initial_episodes

    def raw_episode_conversion(self,
                               replay_buffer_raw: ReplayBuffer,
                               replay_buffer_output: dict,
                               episode_name: str,
                               shape_meta: dict):
        # ep: 'episode_xx'
        replay_buffer_output["data"][episode_name] = dict()
        raw_to_obs(
            replay_buffer_raw["data"][episode_name], replay_buffer_output["data"][episode_name], shape_meta,
            raw_policy_timestamp=self.use_raw_policy_timestamps
        )
        if self.action_type == "pose3":
            raw_to_action3(
                replay_buffer_raw["data"][episode_name],
                replay_buffer_output["data"][episode_name],
                self.id_list,
            )
        elif self.action_type == "pose9":
            raw_to_action9(
                replay_buffer_raw["data"][episode_name],
                replay_buffer_output["data"][episode_name],
                self.id_list,
                shape_meta,
            )
        elif self.action_type == "pose9wrench6":
            raw_to_action15(
                replay_buffer_raw["data"][episode_name],
                replay_buffer_output["data"][episode_name],
                self.id_list,
                shape_meta,
            )
        elif self.action_type == "pose9wrench6g1":
            raw_to_action16(
                replay_buffer_raw["data"][episode_name],
                replay_buffer_output["data"][episode_name],
                self.id_list,
                shape_meta,
            )
        elif self.action_type == "pose9pose9s1":
            raw_to_action19(
                replay_buffer_raw["data"][episode_name],
                replay_buffer_output["data"][episode_name],
                self.id_list,
            )
        elif self.action_type == "pose9pose9s1g2":
            raw_to_action21(
                replay_buffer_raw["data"][episode_name],
                replay_buffer_output["data"][episode_name],
                self.id_list,
            )
        else:
            raise RuntimeError("unsupported")
        # print(f"episode {ep}")
        # print("raw_pose:", replay_buffer_raw['data'][ep]["ts_pose_command_0"][:])
        # print("raw_policy:", replay_buffer_raw['data'][ep]["policy_pose_command_0"][:])
        # print("processed:", replay_buffer['data'][ep]["action"])


    def raw_episodes_conversion(
        self, replay_buffer_raw: ReplayBuffer, shape_meta: dict
    ):
        replay_buffer = dict()
        replay_buffer["data"] = dict()

        assert replay_buffer_raw is not None, "replay_buffer_raw is None"
        for ep in replay_buffer_raw["data"].keys():
            # iterates over episodes
            self.raw_episode_conversion(replay_buffer_raw, replay_buffer, ep, shape_meta)

        # meta
        replay_buffer["meta"] = replay_buffer_raw["meta"]
        return replay_buffer

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        if self.action_type == "pose3":
            action_to_action_sample = action3_to_action_sample
        elif self.action_type == "pose9":
            action_to_action_sample = action9_to_action_sample
        elif self.action_type == "pose9wrench6":
            action_to_action_sample = action15_to_action_sample
        elif self.action_type == "pose9wrench6g1":
            action_to_action_sample = action16_to_action_sample
        elif self.action_type == "pose9pose9s1":
            action_to_action_sample = action19_to_action_sample
        elif self.action_type == "pose9pose9s1g2":
            action_to_action_sample = action21_to_action_sample
        else:
            raise RuntimeError("unsupported")

        val_set.sampler = SequenceSampler(
            shape_meta=self.shape_meta,
            replay_buffer=self.replay_buffer_dict,
            sparse_query_frequency_down_sample_steps=self.sparse_query_frequency_down_sample_steps,
            episode_mask=self.val_mask,
            action_padding=self.action_padding,
            obs_to_obs_sample=obs_to_obs_sample,
            action_to_action_sample=action_to_action_sample,
            id_list=self.id_list,
            weighted_sampling=self.weighted_sampling,
            correction_horizon=self.correction_horizon,
            detect_correction_with_wrench = True,
            correction_force_threshold=self.correction_force_threshold,
            correction_torque_threshold=self.correction_torque_threshold,
            use_raw_policy_timestamps=self.use_raw_policy_timestamps,
        )
        val_set.val_mask = ~self.val_mask
        return val_set

    def get_normalizer(self, **kwargs) -> tuple:
        """Compute normalizer for each key in the dataset.
        Note: only low_dim and action are considered. Image does not need normalization.
        return: tuple of normalizers for sparse and dense obs. Dense one might be None.
        """
        sparse_normalizer = LinearNormalizer()

        # gather all data in cache
        self.sampler.ignore_rgb(True)
        dataloader = torch.utils.data.DataLoader(
            dataset=self,
            batch_size=64,
            num_workers=32,
        )

        data_cache_sparse = {}
        data_cache_dense = {}

        for batch in tqdm(dataloader, desc="iterating dataset to get normalization"):
            # sparse obs
            for key in self.shape_meta["sample"]["obs"]["sparse"].keys():
                if self.shape_meta["obs"][key]["type"] == "low_dim" or self.shape_meta["obs"][key]["type"] == "timestamp":
                    if key not in data_cache_sparse.keys():
                        data_cache_sparse[key] = []
                    data_cache_sparse[key].append(
                        copy.deepcopy(batch["obs"]["sparse"][key])
                    )
            if "action" not in data_cache_sparse.keys():
                data_cache_sparse["action"] = []
            data_cache_sparse["action"].append(copy.deepcopy(batch["action"]["sparse"]))
        self.sampler.ignore_rgb(False)

        # concatenate all data
        for key in data_cache_sparse.keys():
            # data[key] = (# batches, B, T, D)
            data_cache_sparse[key] = np.concatenate(data_cache_sparse[key])  # (B, T, D)
            if not self.temporally_independent_normalization:
                data_cache_sparse[key] = rearrange(
                    data_cache_sparse[key], "B T ... -> (B T) (...)"
                )  # (B*T, D)

        # sparse: compute normalizer for action
        sparse_action_normalizers = list()
        print("data_cache_sparse['action']", data_cache_sparse["action"].shape)

        if self.action_type == "pose3":
            AL = 3
        elif self.action_type == "pose9":
            AL = 9
        elif self.action_type == "pose9wrench6":
            AL = 15
        elif self.action_type == "pose9wrench6g1":
            AL = 16
        elif self.action_type == "pose9pose9s1":
            AL = 19
        elif self.action_type == "pose9pose9s1g2":
            AL = 21
        else:
            raise RuntimeError("unsupported")

        for i in range(len(self.id_list)):
            sparse_action_normalizers.append(
                get_range_normalizer_from_stat(
                    array_to_stats(
                        data_cache_sparse["action"][..., i * AL + 0 : i * AL + 3]
                    )
                )
            )  # pos
            if AL < 9:
                continue
            sparse_action_normalizers.append(
                get_identity_normalizer_from_stat(
                    array_to_stats(
                        data_cache_sparse["action"][..., i * AL + 3 : i * AL + 9]
                    )
                )
            )  # rot
            if AL < 15:
                continue
            if AL == 15 or AL == 16:
                # this only exists for AL = 15 or 16
                # always normalize wrench if in action
                sparse_action_normalizers.append(
                    get_range_normalizer_from_stat(
                        array_to_stats(
                            data_cache_sparse["action"][..., i * AL + 9 : i * AL + 15]
                        )
                    )
                ) # wrench
            if AL == 16:
                sparse_action_normalizers.append(
                    get_range_normalizer_from_stat(
                        array_to_stats(
                            data_cache_sparse["action"][..., i * AL + 15 : i * AL + 16]
                        )
                    )
                )
            if AL < 19:
                continue
            sparse_action_normalizers.append(
                get_range_normalizer_from_stat(
                    array_to_stats(
                        data_cache_sparse["action"][..., i * AL + 9 : i * AL + 12]
                    )
                )
            )  # pos
            sparse_action_normalizers.append(
                get_identity_normalizer_from_stat(
                    array_to_stats(
                        data_cache_sparse["action"][..., i * AL + 12 : i * AL + 18]
                    )
                )
            )  # rot
            sparse_action_normalizers.append(
                get_range_normalizer_from_stat(
                    array_to_stats(
                        data_cache_sparse["action"][..., i * AL + 18 : i * AL + 19]
                    )
                )
            )  # stiffness
            if AL < 21:
                continue
            sparse_action_normalizers.append(
                get_range_normalizer_from_stat(
                    array_to_stats(
                        data_cache_sparse["action"][..., i * AL + 19 : i * AL + 20]
                    )
                )
            )  # gripper
            sparse_action_normalizers.append(
                get_range_normalizer_from_stat(
                    array_to_stats(
                        data_cache_sparse["action"][..., i * AL + 20 : i * AL + 21]
                    )
                )
            )  # grasping force

        sparse_normalizer["action"] = concatenate_normalizer(sparse_action_normalizers)

        # sparse: compute normalizer for obs
        for key in self.shape_meta["sample"]["obs"]["sparse"].keys():
            type = self.shape_meta["obs"][key]["type"]
            if type == "low_dim":
                stat = array_to_stats(data_cache_sparse[key])
                if "eef_pos" in key:
                    this_normalizer = get_range_normalizer_from_stat(stat)
                elif "rot_axis_angle" in key:
                    this_normalizer = get_identity_normalizer_from_stat(stat)
                elif "wrench" in key:
                    if self.normalize_wrench:
                        this_normalizer = get_range_normalizer_from_stat(stat)
                    else:
                        this_normalizer = get_identity_normalizer_from_stat(stat)
                elif "gripper" in key:
                    this_normalizer = get_range_normalizer_from_stat(stat)
                else:
                    raise RuntimeError("unsupported")
                sparse_normalizer[key] = this_normalizer
            elif type == "rgb":
                sparse_normalizer[key] = get_image_identity_normalizer()
            elif type == "timestamp":
                stat = array_to_stats(data_cache_sparse[key])
                sparse_normalizer[key] = get_identity_normalizer_from_stat(stat)
            else:
                raise RuntimeError("unsupported")

        return sparse_normalizer, None

    ### Load a newly saved episode from file, load it into memory store (replaybuffer),
    ### and update the directory store.
    def load_new_episode(self, episode_name: str, ft_sensor_configuration: str):
        ## 1. read it into directory store, which also save the zarr file
        output_dir = self.dataset_path + "processed"
        episode_config = {
            "input_dir": self.dataset_path + "raw",
            "output_dir": output_dir,
            "id_list": self.id_list,
            "ft_sensor_configuration": ft_sensor_configuration,
            "num_threads": 10,
            "has_correction": True,
            "save_video": False,
            "max_workers": 16, # used in EpisodeDataBuffer
        }
        process_one_episode_into_zarr(episode_name, self.ds_root, episode_config)
        generate_meta_for_zarr(self.ds_root, episode_config)
        ## Now:
        ##      zarr file:  has new ep raw data, new meta
        ##      ds_root:    has new ep raw data, new meta

        ## 2. Generate VT labels, make it raw data
        compute_vt_for_episode(episode_name, self.ds_root["data"][episode_name], self.virtual_target_config)
        ## Now:
        ##      zarr file:  has new ep raw data, new meta, vt
        ##      ds_root:    has new ep raw data, new meta, vt
        
        ## 3. copy the new episode (raw) from directory store into memory store
        this_path = '/data/' + episode_name
        memory_store_root = self.replay_buffer_raw.root
        memory_store = self.replay_buffer_raw.root.store
        memory_store_root.require_group("data", overwrite=False)
        n_copied, n_skipped, n_bytes_copied = zarr.copy_store(
            source=self.ds_root.store, dest=memory_store,
            source_path=this_path, dest_path=this_path,
            if_exists="raise"
        )
        n_copied, n_skipped, n_bytes_copied = zarr.copy_store(
            source=self.ds_root.store, dest=memory_store,
            source_path='/meta/', dest_path='/meta/',
            if_exists="replace"
        )
        ## Now:
        ##      zarr file:    has new ep raw data, new meta, vt
        ##      ds_root:      has new ep raw data, new meta, vt
        ##      memory store: has new ep raw data, new meta, vt
        
        ## 4. Then run raw_conversion, save the result into replay_buffer_dict
        self.raw_episode_conversion(self.replay_buffer_raw, self.replay_buffer_dict, episode_name, self.shape_meta)
        self.replay_buffer_dict["meta"] = self.replay_buffer_raw["meta"]
        ## Now:
        ##      zarr file:          has new ep raw data, new meta, vt
        ##      ds_root:            has new ep raw data, new meta, vt
        ##      memory store:       has new ep raw data, new meta, vt
        ##      replay_buffer_dict: has new ep obs, action, new meta

    
    def load_new_episodes(self, episode_names: list, ft_sensor_configuration: str):
        for name in episode_names:
            self.load_new_episode(name, ft_sensor_configuration)

        self.val_mask = get_val_mask(
            n_episodes=self.replay_buffer_raw.n_episodes, val_ratio=self.val_ratio, seed=self.seed
        )
        train_mask = ~self.val_mask

        # recreate the sampler
        self.sampler = SequenceSampler(
            shape_meta=self.shape_meta,
            replay_buffer=self.replay_buffer_dict,
            sparse_query_frequency_down_sample_steps=self.sparse_query_frequency_down_sample_steps,
            episode_mask=train_mask,
            action_padding=self.action_padding,
            obs_to_obs_sample=obs_to_obs_sample,
            action_to_action_sample=self.action_to_action_sample,
            id_list=self.id_list,
            weighted_sampling=self.weighted_sampling,
            correction_horizon=self.correction_horizon,
            detect_correction_with_wrench=True,
            new_episode_prob=self.new_episode_prob,
            num_new_episodes=len(episode_names),
            correction_force_threshold=self.correction_force_threshold,
            correction_torque_threshold=self.correction_torque_threshold,
            num_initial_episodes=self.num_initial_episodes,
            use_raw_policy_timestamps=self.use_raw_policy_timestamps,
        )
        self.sampling_weights = self.sampler.sample_weights


    def __len__(self):
        return len(self.sampler)

    # @profile
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if not self.threadpool_limits_is_applied:
            threadpool_limits(1)
            self.threadpool_limits_is_applied = True
        obs_dict, action_array = self.sampler.sample_sequence(idx)

        torch_data = {
            "obs": dict_apply(obs_dict, torch.from_numpy),
            "action": dict_apply(action_array, torch.from_numpy),
        }
        return torch_data

    def num_of_episodes(self):
        return self.replay_buffer_raw.n_episodes