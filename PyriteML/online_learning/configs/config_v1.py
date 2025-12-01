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
    "slow_down_factor": 1,  # set to 2 to slow down the execution by 2x. Does not affect data saving rate
    "sparse_execution_horizon": 12,  # execution horizon for the base policy
    "delay_tolerance_s": 0.3, # delay larger than this will trigger termination
    "max_duration_s": 3500, # actor will quit after this long
    "test_nominal_target": False,
    "translational_stiffness": [1000, 1000, 1000],
    "rotational_stiffness": 70, # or 25 for belt task
    "send_transitions_to_server": True, # if False, actor will not send robot data to the learner server. Useful for evaluation
    "no_visual_mode": False,
    "device": "cuda",
    # below are debugging options. Disabled by default
    "fix_orientation": False,
    "scale_and_cap_residual_action": False,
    "residual_action_scale_ratio": 1.0,  # ratio of residual to nominal action
}

hardware_para = {
    "hardware_config_path": hardware_config_folder_path + "/belt_assembly.yaml",
}

learner_para = {
    # if residual_ckpt_path is specified and points to a valid checkpoint, the residual learner will load the checkpoint
    # Note that currently the new checkpoints will still be saved to a new folder
    "residual_ckpt_path": None,
    # "residual_ckpt_path": "/2025.05.13_14.21.08_belt_residual_no_base_action_online_residual_mlp",

    "num_episodes_before_first_training": 50, # start training after a good number of episodes collected
    
    # Below are parameters for multi-batch training.
    # These features are not used in the paper.
    "num_of_initial_episodes": 0, # first n episodes, both correction and no correction data are used
    "num_of_new_episodes": 0, # last n episodes to sample 50% from. set to 0 to disable
}

# online_learning_para needs to be the same across learner and actor
online_learning_para = {
    "data_folder_path": data_folder_path + "/online_belt/", # where to save online correction data
    "policy_workspace_config_name": "train_online_residual_mlp_workspace", # workspace config name for the residual policy
    "transformers": False,
    "network_weight_topic": "network_weights_topic",
    "transitions_topic": "transitions_topic",
    "network_weight_expire_time_s": 3500, # time after which the actor-learner communication is considered lost
    "transitions_topic_expire_time_s": 3500, # time after which the actor-learner communication is considered lost
}

if run_learner_on_server:
    online_learning_para["network_server_endpoint"] = "tcp://localhost:18889"
    online_learning_para["transitions_server_endpoint"] = "tcp://localhost:18888"
else:
    # IPC is a lot faster than tcp
    online_learning_para["network_server_endpoint"] = "ipc:///tmp/feeds/2"
    online_learning_para["transitions_server_endpoint"] = "ipc:///tmp/feeds/3"