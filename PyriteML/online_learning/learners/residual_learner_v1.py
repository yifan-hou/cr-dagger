import sys
import os
from typing import Dict, Callable, Tuple, List
import pathlib

SCRIPT_PATH = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(SCRIPT_PATH, "../../"))

import numpy as np
import torch
import dill
import hydra
import time
import os
import hydra
import torch
from omegaconf import OmegaConf
import pathlib
import numpy as np
import matplotlib.pyplot as plt

from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.workspace.train_online_residual_mlp_workspace import TrainOnlineResidualMLPWorkspace 
from diffusion_policy.dataset.dynamic_dataset import DynamicDataset
OmegaConf.register_new_resolver("eval", eval, replace=True)

from online_learning.learner import Learner

if "PYRITE_CHECKPOINT_FOLDERS" not in os.environ:
    raise ValueError("Please set the environment variable PYRITE_CHECKPOINT_FOLDERS")
checkpoint_folder_path = os.environ.get("PYRITE_CHECKPOINT_FOLDERS")
if "PYRITE_DATASET_FOLDERS" not in os.environ:
    raise ValueError("Please set the environment variable PYRITE_DATASET_FOLDERS")
data_folder_path = os.environ.get("PYRITE_DATASET_FOLDERS")

from PyriteML.online_learning.configs.config_v1 import online_learning_para, learner_para

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.parent.joinpath(
        'diffusion_policy','config')),
    config_name=online_learning_para["policy_workspace_config_name"]
)
def main(residual_ws_cfg: OmegaConf):
    print("Starting residual learner")
    learner_node = Learner(
        network_server_endpoint=online_learning_para["network_server_endpoint"],
        network_weight_topic=online_learning_para["network_weight_topic"],
        transitions_server_endpoint=online_learning_para["transitions_server_endpoint"],
        transitions_topic=online_learning_para["transitions_topic"],
        network_weight_expire_time_s=online_learning_para["network_weight_expire_time_s"],
    )

    OmegaConf.resolve(residual_ws_cfg)
    residual_cls = hydra.utils.get_class(residual_ws_cfg._target_)
    residual_workspace: BaseWorkspace = residual_cls(residual_ws_cfg)
    
    # load residual policy workspace
    print("Creating residual policy workspace according to config: ", online_learning_para["policy_workspace_config_name"])
    ## load previous checkpoint file if exists
    if learner_para["residual_ckpt_path"] is not None and os.path.exists(checkpoint_folder_path + learner_para["residual_ckpt_path"]):
        print("--------------------------------------------------------------------------")
        print("[Learner] Found residual policy checkpoint. Loading into workspace")
        print("--------------------------------------------------------------------------")
        residual_payload = torch.load(open(pathlib.Path(checkpoint_folder_path + learner_para["residual_ckpt_path"]).joinpath('checkpoints', 'latest.ckpt'), "rb"), map_location="cpu", pickle_module=dill)
        residual_workspace.load_payload(residual_payload, exclude_keys=None, include_keys=None)
        # after loading a checkpoint, send the weights immediately
        payloads = {
            "model": residual_workspace.model,
            "trainable_obs_encoders": residual_workspace.trainable_obs_encoders,
            "sparse_normalizer": residual_workspace.sparse_normalizer
        }
        learner_node.send_network_weights(payloads)
    else:
        print("--------------------------------------------------------------------------")
        print("[Learner] No residual policy checkpoint found. Starting from scratch")
        print("--------------------------------------------------------------------------")

    # load dataset
    dataset: DynamicDataset
    # residual_ws_cfg.task.dataset.num_initial_episodes = learner_para["num_episodes_before_first_training"]
    dataset = hydra.utils.instantiate(residual_ws_cfg.task.dataset)
    dataset.load_initial_dataset(online_learning_para["data_folder_path"],
                                 learner_para["num_of_initial_episodes"],
                                 learner_para["num_of_new_episodes"])

    traning_preparation_done = False
    got_new_episodes = False
    
    # main learner loop, one episode per loop
    while True:
        # train
        if dataset.num_of_episodes() < learner_para["num_episodes_before_first_training"] or (residual_workspace.epoch > residual_ws_cfg.training.first_time_num_epochs and got_new_episodes == False):
            print("[Learner] Waiting for enough episodes to train. Current number of episodes: ", dataset.num_of_episodes())
        else:
            print("[Learner] Training with ", dataset.num_of_episodes(), " episodes")
            if not traning_preparation_done:
                # compute normalizers, etc
                residual_workspace.train_preparation(dataset)
                traning_preparation_done = True


            #  train
            residual_workspace.run(dataset)

            #  send new weights
            payloads = {
                "model": residual_workspace.model,
                "trainable_obs_encoders": residual_workspace.trainable_obs_encoders,
                "sparse_normalizer": residual_workspace.sparse_normalizer
            }
            learner_node.send_network_weights(payloads)
        

        # Receive new episode name
        all_new_episode_names = []
        episode_data = None
        while True:
            print("[Learner] Waiting for new episode data ...")
            try:
                episode_data = learner_node.receive_transitions()
            except Exception as e:
                print("[Learner] Error: ", e)
            
            if episode_data is not None:
                episode_name = episode_data["episode_name"]
                print("[Learner] Received new episode data: ", episode_name)
                all_new_episode_names.append(episode_name)
                got_new_episodes = True
            
            if len(all_new_episode_names) > 0 and episode_data is None:
                break
        
            time.sleep(1)

        print(f"[Learner] Loading {len(all_new_episode_names)} new episodes ...")
        dataset.load_new_episodes(  # this will also save the data to the zarr file
            episode_names = all_new_episode_names,
            ft_sensor_configuration = "handle_on_robot"
        )

if __name__ == "__main__":
    main()
