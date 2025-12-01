import os

if "PYRITE_HARDWARE_CONFIG_FOLDERS" not in os.environ:
    raise ValueError(
        "Please set the environment variable PYRITE_HARDWARE_CONFIG_FOLDERS"
    )
if "PYRITE_DATASET_FOLDERS" not in os.environ:
    raise ValueError("Please set the environment variable PYRITE_DATASET_FOLDERS")
if "PYRITE_CONTROL_LOG_FOLDERS" not in os.environ:
    raise ValueError("Please set the environment variable PYRITE_CONTROL_LOG_FOLDERS")

hardware_config_folder_path = os.environ.get("PYRITE_HARDWARE_CONFIG_FOLDERS")
data_folder_path = os.environ.get("PYRITE_DATASET_FOLDERS")

run_learner_on_server = False

control_para = {
    "raw_time_step_s": 0.002,  # dt of raw data collection. Used to compute time step from time_s such that the downsampling according to shape_meta works.
    "slow_down_factor": 1,  # 3 for flipup, 1.5 for wiping
    "sparse_execution_horizon": 12,  # 12 for flipup, 8/24 for wiping
    "delay_tolerance_s": 0.3, # delay larger than this will trigger termination
    "max_duration_s": 3500,
    "test_nominal_target": False,
    "translational_stiffness": [1000, 1000, 1000],
    "rotational_stiffness": 70, # or 25 for belt task
    "send_transitions_to_server": False,
    "fix_orientation": False,
    "no_visual_mode": False,
    "device": "cuda",
    "scale_and_cap_residual_action": False,
    "residual_action_scale_ratio": 0.3,  # ratio of residual to nominal action
}

hardware_para = {
    "hardware_config_path": hardware_config_folder_path + "/belt_assembly.yaml",
}

learner_para = {
    # if residual_ckpt_path is specified and points to a valid checkpoint, the residual learner will load the checkpoint
    # However, currently the new checkpoints will still be saved to a new folder
    "residual_ckpt_path": None,
    # force control
    # "residual_ckpt_path": "/2025.04.23_17.56.55_stow_residual_online_residual_mlp",
    # "residual_ckpt_path": "/2025.04.24_12.21.14_stow_residual_online_residual_mlp",
    # "residual_ckpt_path": "/2025.04.24_16.09.13_stow_residual_online_residual_mlp",
    # 50 episodes
    # "residual_ckpt_path": "/2025.04.25_00.56.38_stow_residual_online_residual_mlp",
    # "residual_ckpt_path": "/2025.04.25_19.11.47_stow_residual_online_residual_mlp",
    # transformer vanilla
    # "residual_ckpt_path": "/2025.04.25_08.15.26_stow_residual_online_residual_tf",
    # "residual_ckpt_path": "/2025.04.25_08.13.21_stow_residual_online_residual_tf", # with time encoding
    # "residual_ckpt_path": "/2025.04.26_03.26.27_stow_residual_online_residual_tf",
    # ablation, no force
    # "residual_ckpt_path": "/2025.04.30_04.55.36_stow_residual_online_residual_mlp",
    # no force, take over data
    # "residual_ckpt_path": "/2025.04.30_20.08.53_stow_residual_online_residual_mlp",
    # ablation, no sampling more
    # "residual_ckpt_path": "/2025.04.26_07.12.23_stow_residual_online_residual_mlp",
    # ablation, sample before and after correction
    # "residual_ckpt_path": "/2025.04.30_03.09.57_stow_residual_online_residual_mlp",
    # residual w take over correction
    # "residual_ckpt_path": "/2025.04.29_02.11.23_stow_residual_online_residual_mlp",

    # 1st batch
    # "residual_ckpt_path": "/2025.04.14_14.59.08_stow_residual_online_residual_mlp",
    # "residual_ckpt_path": "/2025.04.15_15.29.49_stow_residual_online_residual_mlp",
    # "residual_ckpt_path": "/2025.04.22_14.21.57_stow_residual_online_residual_mlp",
    # "residual_ckpt_path": "/2025.04.22_22.32.28_stow_residual_online_residual_mlp",
    # multi-modal
    # "residual_ckpt_path": "/2025.04.18_18.09.40_stow_residual_online_residual_mlp",
    # "residual_ckpt_path": "/2025.04.19_11.50.55_stow_residual_online_residual_mlp",
    # 2nd batch
    # "residual_ckpt_path": "/2025.04.20_23.03.58_stow_residual_online_residual_mlp",
    # "residual_ckpt_path": "/2025.04.21_16.17.18_stow_residual_online_residual_mlp",

    # residual online batch 20
    # "residual_ckpt_path": "/2025.04.30_16.24.56_stow_residual_online_residual_mlp",
    # residual online batch 20-10
    # "residual_ckpt_path": "/2025.04.30_17.06.36_stow_residual_online_residual_mlp",
    # residual online batch 20-10-10
    # "residual_ckpt_path": "/2025.04.30_17.41.55_stow_residual_online_residual_mlp",
    # residual online batch 20-10-10-10
    # "residual_ckpt_path": "/2025.04.30_18.13.49_stow_residual_online_residual_mlp",

    # "residual_ckpt_path": "/2025.05.01_22.44.45_stow_residual_online_residual_mlp",

    # residual no force no base action delta correction
    # "residual_ckpt_path": "/2025.05.07_06.16.00_belt_residual_no_base_action_no_force_online_residual_mlp",

    # residual no force no base action takeover
    # "residual_ckpt_path": "/2025.05.07_07.10.40_belt_residual_no_base_action_no_force_online_residual_mlp",

    # (no base action as residual input) residual with force delta correction
    # "residual_ckpt_path": "/2025.05.07_07.15.16_belt_residual_no_base_action_online_residual_mlp_wo_base_action_shelf",

    # (no base action as residual input) residual with force takeover
    # "residual_ckpt_path": "/2025.05.07_07.13.56_belt_residual_no_base_action_online_residual_mlp_wo_base_action_shelf",

    ######## ------------------ Belt Assembly ------------------ ########
    # "residual_ckpt_path": "/2025.05.02_01.15.42_stow_residual_online_residual_mlp",
    # "residual_ckpt_path": "/2025.05.01_23.13.19_stow_residual_online_residual_mlp",
    # 90, 4x10
    # "residual_ckpt_path": "/2025.05.03_02.51.57_stow_residual_online_residual_mlp",
    # 90, 3x20
    # "residual_ckpt_path": "/2025.05.03_02.42.22_stow_residual_online_residual_mlp",
    # correction only, 90+15
    # "residual_ckpt_path": "/2025.05.02_22.21.14_belt_residual_online_residual_mlp",
    # correction only, no base action, 90+15
    # "residual_ckpt_path": "/2025.05.03_14.18.18_belt_residual_no_base_action_online_residual_mlp",
    # new data
    # "residual_ckpt_path": "/2025.05.03_23.50.53_belt_residual_no_base_action_online_residual_mlp",
    # v4 50
    # "residual_ckpt_path": "/2025.05.04_12.58.33_belt_residual_no_base_action_online_residual_mlp",
    # "residual_ckpt_path": "/2025.05.04_15.49.50_belt_residual_no_base_action_online_residual_mlp",
    # v6 single batch 50+21
    # "residual_ckpt_path": "/2025.05.05_02.40.51_belt_residual_no_base_action_online_residual_mlp",
    # 50 + 30 2nd batch
    # "residual_ckpt_path": "/2025.05.04_20.35.47_belt_residual_no_base_action_online_residual_mlp",

    # v5 50 more samples after correction (longer horizon )
    # "residual_ckpt_path": "/2025.05.12_19.48.14_belt_residual_no_base_action_online_residual_mlp_belt",
    # "residual_ckpt_path": "/2025.05.12_19.58.22_belt_residual_no_base_action_online_residual_mlp_belt",
    # "residual_ckpt_path": "/2025.05.12_22.19.39_belt_residual_no_base_action_online_residual_mlp",
    # "residual_ckpt_path": "/2025.05.13_14.21.08_belt_residual_no_base_action_online_residual_mlp",

    "num_episodes_before_first_training": 5000,
    "num_of_initial_episodes": 0, # first n episodes, both correction and no correction data are used
    "num_of_new_episodes": 0, # last n episodes to sample 50% from. set to 0 to disable
}

# online_learning_para needs to be the same across learner and actor
online_learning_para = {
    "data_folder_path": data_folder_path + "/online_belt_v9/",
    # "data_folder_path": data_folder_path + "/online_stow_nb_v5_50/",
    "policy_workspace_config_name": "train_online_residual_mlp_workspace",
    # "policy_workspace_config_name": "train_online_residual_mlp_workspace_no_force",
    # "policy_workspace_config_name": "train_online_residual_tf_workspace",
    "transformers": False,
    "network_weight_topic": "network_weights_topic",
    "transitions_topic": "transitions_topic",
    "network_weight_expire_time_s": 1200,
    "transitions_topic_expire_time_s": 1200,
}
if run_learner_on_server:
    online_learning_para["network_server_endpoint"] = "tcp://localhost:18889"
    online_learning_para["transitions_server_endpoint"] = "tcp://localhost:18888"
else:
    online_learning_para["network_server_endpoint"] = "ipc:///tmp/feeds/2"
    online_learning_para["transitions_server_endpoint"] = "ipc:///tmp/feeds/3"