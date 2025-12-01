import sys
import os

SCRIPT_PATH = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(SCRIPT_PATH, "../../"))

import cv2
import numpy as np
import torch
import hydra
import time
import matplotlib.pyplot as plt
import zarr
import shutil
import copy


from PyriteEnvSuites.envs.task.manip_server_handle_env import ManipServerHandleEnv

from PyriteEnvSuites.utils.env_utils import ts_to_js_traj, pose9g1_to_traj, pose9pose9s1_to_traj, get_real_obs_resolution, decode_stiffness

from PyriteConfig.tasks.common.common_type_conversions import raw_to_obs
from PyriteUtility.spatial_math import spatial_utilities as su
from PyriteUtility.planning_control.mpc import ModelPredictiveControllerHybrid
from PyriteUtility.planning_control.trajectory import LinearTransformationInterpolator
from PyriteUtility.pytorch_utils.model_io import load_policy
from PyriteUtility.plotting.matplotlib_helpers import set_axes_equal
from PyriteUtility.plotting.traj_and_rgb import draw_timed_traj_and_rgb
from PyriteUtility.common import GracefulKiller
from PyriteUtility.common import dict_apply
from PyriteConfig.tasks.common import common_type_conversions as task
from PyriteML.diffusion_policy.workspace.base_workspace import BaseWorkspace
from PyriteML.online_learning.actor import Actor

from PyriteML.online_learning.configs.config_v1 import control_para, hardware_para, online_learning_para

policy_workspace_config_path = "../../PyriteML/diffusion_policy/config/"

def scale_and_cap_residual_action(r, scale = 1, translation_limit=0.05, rotation_limit=0.4):
    """
    Cap the residual action to be within the specified limits. keep the direction but scale down the magnitude.
    :param r: residual action (SE3)
    :param translation_limit: limit for translation
    :param rotation_limit: limit for rotation
    :return: capped residual action
    """
    r_spt = su.SE3_to_spt(r) * scale

    if np.linalg.norm(r_spt[:3]) > translation_limit:
        r_spt[:3] = r_spt[:3] / np.linalg.norm(r_spt[:3]) * translation_limit
    if np.linalg.norm(r_spt[3:6]) > rotation_limit:
        r_spt[3:6] = r_spt[3:6] / np.linalg.norm(r_spt[3:6]) * rotation_limit
    return su.spt_to_SE3(r_spt)

def print_env_status(env):
    obs_raw = env.get_observation_from_buffer()
    
    for id in env.id_list:
        robot_wrench = obs_raw[f"robot_wrench_{id}"]
        ati_wrench = obs_raw[f"wrench_{id}"]
        robot_wrench_average = np.mean(robot_wrench, axis=0)
        ati_wrench_average = np.mean(ati_wrench, axis=0)
        print(f"[Env Status] robot_wrench_{id} average: ", robot_wrench_average)
        print(f"[Env Status] ati_wrench_{id} average: ", ati_wrench_average)

def main():
    # create the actor
    actor_node = Actor(
        network_server_endpoint=online_learning_para["network_server_endpoint"],
        network_weight_topic=online_learning_para["network_weight_topic"],
        transitions_server_endpoint=online_learning_para["transitions_server_endpoint"],
        transitions_topic=online_learning_para["transitions_topic"],
        transitions_topic_expire_time_s=online_learning_para["transitions_topic_expire_time_s"],
    )

    # load residual policy workspace
    print("Loading residual policy workspace")
    device = torch.device(control_para["device"])
    with hydra.initialize(version_base=None,
                          config_path=policy_workspace_config_path):
        residual_cfg = hydra.compose(config_name=online_learning_para["policy_workspace_config_name"])

    # print(OmegaConf.to_yaml(residual_cfg))
    residual_cls = hydra.utils.get_class(residual_cfg._target_)
    residual_workspace = residual_cls(residual_cfg)

    residual_workspace: BaseWorkspace
    residual_shape_meta = residual_cfg.task.shape_meta
    residual_weights_loaded = False # only set to true if received weights from learner.

    use_transformers = online_learning_para["transformers"]

    # load the base policy specified in the residual policy config
    print("Loading base policy: ", residual_cfg.base_policy_ckpt)
    policy, shape_meta = load_policy(residual_cfg.base_policy_ckpt, device)

    # image size
    (image_width, image_height) = get_real_obs_resolution(shape_meta)

    base_rgb_query_size = (
        shape_meta["sample"]["obs"]["sparse"]["rgb_0"]["horizon"] - 1
    ) * shape_meta["sample"]["obs"]["sparse"]["rgb_0"]["down_sample_steps"] + 1
    base_ts_pose_query_size = (
        shape_meta["sample"]["obs"]["sparse"]["robot0_eef_pos"]["horizon"] - 1
    ) * shape_meta["sample"]["obs"]["sparse"]["robot0_eef_pos"]["down_sample_steps"] + 1
    if "policy_robot0_eef_pos" in residual_shape_meta["sample"]["obs"]["sparse"]:
        residual_base_action_query_size = (
            residual_shape_meta["sample"]["obs"]["sparse"]["policy_robot0_eef_pos"]["horizon"] - 1
        ) * residual_shape_meta["sample"]["obs"]["sparse"]["policy_robot0_eef_pos"]["down_sample_steps"] + 1
    else:
        residual_base_action_query_size = base_ts_pose_query_size
    if "robot0_eef_wrench" in residual_shape_meta["sample"]["obs"]["sparse"]:
        wrench_query_size = (
            residual_shape_meta["sample"]["obs"]["sparse"]["robot0_eef_wrench"]["horizon"] - 1
        ) * residual_shape_meta["sample"]["obs"]["sparse"]["robot0_eef_wrench"][
            "down_sample_steps"
        ] + 1
    else:
        wrench_query_size = (
            32 - 1
        ) * 4 + 1
    
    query_sizes_sparse = {
        "rgb": base_rgb_query_size,
        "ts_pose_fb": max(base_ts_pose_query_size, residual_base_action_query_size),
        "wrench": wrench_query_size,
    }
    query_sizes = {
        "sparse": query_sizes_sparse,
    }

    env = ManipServerHandleEnv(
        camera_res_hw=(image_height, image_width),
        hardware_config_path=hardware_para["hardware_config_path"],
        query_sizes=query_sizes,
        compliant_dimensionality=6,
    )
    env.reset()

    p_timestep_s = control_para["raw_time_step_s"]

    sparse_action_down_sample_steps = shape_meta["sample"]["action"]["sparse"][
        "down_sample_steps"
    ]
    sparse_action_horizon = shape_meta["sample"]["action"]["sparse"]["horizon"]
    sparse_execution_horizon = (
        sparse_action_down_sample_steps * control_para["sparse_execution_horizon"]
    )
    sparse_action_timesteps_s = (
        np.arange(0, sparse_action_horizon)
        * sparse_action_down_sample_steps
        * p_timestep_s
        * control_para["slow_down_factor"]
    )

    residual_action_down_sample_steps = residual_shape_meta["sample"]["action"]["sparse"][
        "down_sample_steps"
    ]
    residual_action_horizon = residual_shape_meta["sample"]["action"]["sparse"]["horizon"]
    residual_action_time_steps_s = (
        np.arange(0, residual_action_horizon)
        * residual_action_down_sample_steps
        * p_timestep_s
        * 1
    )

    action_type = "pose9"  # "pose9" or "pose9pose9s1"
    id_list = [0]
    if shape_meta["action"]["shape"][0] == 9:
        action_type = "pose9"
    elif shape_meta["action"]["shape"][0] == 10:
        action_type = "pose9g1"
    elif shape_meta["action"]["shape"][0] == 15:
        action_type = "pose9wrench6"
    elif shape_meta["action"]["shape"][0] == 19:
        action_type = "pose9pose9s1"
    elif shape_meta["action"]["shape"][0] == 30:
        action_type = "pose9wrench6"
    elif shape_meta["action"]["shape"][0] == 38:
        action_type = "pose9pose9s1"
        id_list = [0, 1]
    else:
        raise RuntimeError("unsupported")

    if action_type == "pose9":
        action_to_trajectory = ts_to_js_traj
    elif action_type == "pose9g1":
        action_to_trajectory = pose9g1_to_traj
    elif action_type == "pose9pose9s1":
        action_to_trajectory = pose9pose9s1_to_traj
    else:
        raise RuntimeError("unsupported")
    
    residual_action_type = "pose9wrench6"
    if residual_shape_meta["action"]["shape"][0] == 9:
        residual_action_type = "pose9"
    elif residual_shape_meta["action"]["shape"][0] == 15:
        residual_action_type = "pose9wrench6"
    elif residual_shape_meta["action"]["shape"][0] == 16:
        residual_action_type = "pose9wrench6g1"
    elif residual_shape_meta["action"]["shape"][0] == 19:
        residual_action_type = "pose9pose9s1"
    elif residual_shape_meta["action"]["shape"][0] == 30:
        residual_action_type = "pose9wrench6"
    elif residual_shape_meta["action"]["shape"][0] == 38:
        residual_action_type = "pose9pose9s1"
    else:
        raise RuntimeError("unsupported")
    print("residual action type:", residual_action_type)

    # the fixed stiffness matrix action can be computed now
    stiffness_matrix = np.eye(6)
    stiffness_matrix[0, 0] = control_para["translational_stiffness"][0]
    stiffness_matrix[1, 1] = control_para["translational_stiffness"][1]
    stiffness_matrix[2, 2] = control_para["translational_stiffness"][2]
    stiffness_matrix[3:, 3:] *= control_para["rotational_stiffness"]
    stiffness_matrix_all = np.zeros((6, 6 * residual_action_horizon))
    for i in range(residual_action_horizon):
        stiffness_matrix_all[:, 6 * i : 6 * i + 6] = stiffness_matrix
    outputs_ts_stiffnesses = stiffness_matrix_all

    print("[Env Runner] Creating controller.")
    controller = ModelPredictiveControllerHybrid(
        shape_meta=shape_meta,
        id_list=id_list,
        policy=policy,
        action_to_trajectory=action_to_trajectory,
        sparse_execution_horizon=sparse_execution_horizon,
        test_sparse_action=True,
        fix_orientation=control_para["fix_orientation"],
    )
    controller.set_time_offset(env.current_hardware_time_s)

    episode_initial_time_s = env.current_hardware_time_s
    execution_duration_s = (
        sparse_execution_horizon * p_timestep_s * control_para["slow_down_factor"]
    )
    print("[Env Runner] Starting main loop.")

    horizon_count = 0
    ts_pose_initial = []
    obs_raw = env.get_observation_from_buffer()
    for id in id_list:
        ts_pose_initial.append(obs_raw[f"ts_pose_fb_{id}"][-1])
    
    # check how many episodes have been run
    num_episodes = 0
    if not os.path.exists(online_learning_para["data_folder_path"]):
        os.makedirs(online_learning_para["data_folder_path"])
    else:
        data_files = os.listdir(online_learning_para["data_folder_path"]+"raw")
        num_episodes = len(data_files)
    print(f"[Env Runner] Number of episodes: {num_episodes}")
    #########################################
    # main loop starts
    #########################################
    while True:
        input(f"[Env Runner] Press Enter to start episode #{num_episodes}.")
        # check if new network weights are available
        print("[Env Runner] Checking for new network weights.")
        if actor_node.receive_network_weights(residual_workspace):
            residual_normalizer = residual_workspace.sparse_normalizer
            residual_policy = residual_workspace.model
            residual_obs_encoder = residual_workspace.obs_encoder
            residual_policy.eval().to(device)
            residual_obs_encoder.eval().to(device)
            residual_weights_loaded = True
            print("[Env Runner] New network weights received.")
        else:
            print("[Env Runner] No new network weights received.")
        
        # start saving data in ManipServer
        env.start_saving_data_for_a_new_episode(online_learning_para["data_folder_path"]+"raw")
        # store the base policy inference actions for sending to the learner
        base_action_targets_all = {id: [] for id in id_list}
        base_action_grippers_all = {id: [] for id in id_list}
        base_action_timestamps_all = []


        # plotting preparation
        log_for_plot = {}
        log_for_plot["rgb_0"] = []
        log_for_plot["rgb_time_stamps_0"] = []
        log_for_plot["ts_pose_fb_0"] = []
        log_for_plot["robot_time_stamps_0"] = []
        log_for_plot["base_action_0"] = []
        log_for_plot["base_action_time_stamps_0"] = []
        log_for_plot["residual_action_0_ref"] = []
        log_for_plot["residual_action_0_vt"] = []
        log_for_plot["residual_action_0_abs"] = []
        log_for_plot["residual_action_time_stamps_0"] = []
        log_for_plot["residual_action_wrench_0"] = []


        killer = GracefulKiller()
        flag_large_delay = False
        while not killer.kill_now:
            horizon_initial_time_s = env.current_hardware_time_s
            print("Starting new horizon at ", horizon_initial_time_s)

            obs_raw = env.get_observation_from_buffer()
            obs_task = dict()
            raw_to_obs(obs_raw, obs_task, shape_meta)

            assert action_type == "pose9"   # the base policy should infer position only

            # Run inference
            controller.set_observation(obs_task["obs"])
            base_action = controller.compute_sparse_control(device)
            if action_type == "pose9":
                action_sparse_target_mats = base_action    # SE3 absolute
            elif action_type == "pose9g1":
                action_sparse_target_mats, action_sparse_eoats = base_action

            base_policy_action = {}
            for id in id_list:
                base_policy_action[f"policy_pose_command_{id}"] = su.SE3_to_pose7(action_sparse_target_mats[id].reshape([-1, 4, 4]))[: control_para["sparse_execution_horizon"]]    # [base_policy_horizon, 4, 4]
                base_policy_action[f"policy_time_stamps_{id}"] = (obs_raw["robot_time_stamps_0"][-1] + sparse_action_timesteps_s)[: control_para["sparse_execution_horizon"]]*1000.0    # [base_policy_horizon]
                if action_type == "pose9g1":
                    base_policy_action[f"policy_gripper_command_{id}"] = action_sparse_eoats[id].reshape([-1, 1])[: control_para["sparse_execution_horizon"]]

            log_for_plot["base_action_0"].append(base_policy_action["policy_pose_command_0"])
            log_for_plot["base_action_time_stamps_0"].append(np.array(base_policy_action["policy_time_stamps_0"])/1000.0)

            # save the base actions
            for id in id_list:
                base_action_targets_all[id].append(base_policy_action[f"policy_pose_command_{id}"])
                if action_type == "pose9g1":
                    base_action_grippers_all[id].append(base_policy_action[f"policy_gripper_command_{id}"])
            base_action_timestamps_all.append(base_policy_action["policy_time_stamps_0"])

            # residual policy loop
            residual_time_start = time.time()
            while time.time() - residual_time_start < execution_duration_s:
                obs_raw = env.get_observation_from_buffer()
                log_for_plot["rgb_0"].append(copy.copy(obs_raw["rgb_0"][-1])) # obs_raw["rgb_0"][-1] is the latest image
                log_for_plot["rgb_time_stamps_0"].append(obs_raw["rgb_time_stamps_0"][-1])
                log_for_plot["ts_pose_fb_0"].append(obs_raw["ts_pose_fb_0"][-1])
                log_for_plot["robot_time_stamps_0"].append(obs_raw["robot_time_stamps_0"][-1])

                # initialize an empty residual action for plotting
                if "pose9wrench6" in residual_action_type:
                    residual_action = np.zeros((residual_action_horizon, 15 * len(id_list)))
                    for id in id_list:
                        residual_action[:, 15 * id + 3] = 1
                        residual_action[:, 15 * id + 7] = 1
                elif residual_action_type == "pose9":
                    residual_action = np.zeros((residual_action_horizon, 9 * len(id_list)))
                    for id in id_list:
                        residual_action[:, 9 * id + 3] = 1
                        residual_action[:, 9 * id + 7] = 1
                elif residual_action_type == "pose9wrench6g1":
                    residual_action = np.zeros((residual_action_horizon, 16 * len(id_list)))
                    for id in id_list:
                        residual_action[:, 16 * id + 3] = 1
                        residual_action[:, 16 * id + 7] = 1
                elif residual_action_type == "pose9g1":
                    residual_action = np.zeros((residual_action_horizon, 10 * len(id_list)))
                    for id in id_list:
                        residual_action[:, 10 * id + 3] = 1
                        residual_action[:, 10 * id + 7] = 1


                # compute the residual action
                if residual_weights_loaded:
                    # run residual policy
                    # add the base policy action to the observation
                    for key, value in base_policy_action.items():
                        obs_raw[key] = value
                    obs_task = dict()
                    raw_to_obs(obs_raw, obs_task, residual_shape_meta, raw_policy_timestamp=use_transformers)
                    residual_obs_data = {}
                    # sub sampling
                    for key, attr in residual_shape_meta["sample"]["obs"]["sparse"].items():
                        data = obs_task["obs"][key]
                        horizon = attr["horizon"]
                        down_sample_steps = attr["down_sample_steps"]
                        if len(data) < (horizon - 1) * down_sample_steps + 1:
                            print(f"Error: {key} is too short. len = {len(data)}, expecting = {(horizon - 1) * down_sample_steps + 1}.")
                        residual_obs_data[key] = data[
                            -(horizon - 1) * down_sample_steps - 1 :: down_sample_steps
                        ]
                    # convert inputs to relative frame
                    obs_sample_np, _ = task.sparse_obs_to_obs_sample(
                        obs_sparse=residual_obs_data,
                        shape_meta=residual_shape_meta,
                        reshape_mode="reshape",
                        id_list=id_list,
                        ignore_rgb=False,
                    )
                    with torch.no_grad():
                        obs_sample = dict_apply(
                            obs_sample_np, lambda x: torch.from_numpy(x).to(device)
                        )
                        if use_transformers:
                            obs_dict_sparse_without_time = dict()
                            for key in obs_sample:
                                if "time_stamps" not in key:
                                    obs_dict_sparse_without_time[key] = obs_sample[key]
                            nobs_sample = residual_normalizer.normalize(obs_dict_sparse_without_time)
                            for key in obs_sample:
                                if "time_stamps" in key:
                                    nobs_sample[key] = obs_sample[key]
                            nobs_sample = dict_apply(nobs_sample, lambda x: x.to(device))
                            # print shapes of all attributes
                            for key, attr in nobs_sample.items():
                                nobs_sample[key] = attr.unsqueeze(0)    # batch size=1
                            nobs_encode, time_encode = residual_obs_encoder(nobs_sample)
                            nresidual_action = residual_policy.predict_actions(nobs_encode, time_encode, (1, residual_shape_meta["sample"]["action"]["sparse"]["horizon"], residual_shape_meta["action"]["shape"][0]))
                        else:
                            nobs_sample = residual_normalizer.normalize(obs_sample)
                            nobs_sample = dict_apply(nobs_sample, lambda x: x.to(device))
                            # print shapes of all attributes
                            for key, attr in nobs_sample.items():
                                nobs_sample[key] = attr.unsqueeze(0)    # batch size=1
                            nobs_encode = residual_obs_encoder(nobs_sample)
                            nresidual_action = residual_policy(nobs_encode)
                        residual_action = residual_normalizer["action"].unnormalize(nresidual_action).squeeze(0).numpy()    # [residual_action_horizon, 19]
                else:
                    time.sleep(0.02)
                
                # decode relative pose from residual action
                # World (W) -> Base (B) -> Residual (R) -> Virtual Target (Vt)
                SE3_BR = su.pose9_to_SE3(residual_action[..., 0:9])
                if residual_action_type == "pose9wrench6" or residual_action_type == "pose9wrench6g1":
                    residual_action_wrench = - residual_action[..., 9:15]

                    twist_RVt = np.zeros_like(residual_action_wrench)
                    twist_RVt[..., 0] = residual_action_wrench[..., 0] / control_para["translational_stiffness"][0]
                    twist_RVt[..., 1] = residual_action_wrench[..., 1] / control_para["translational_stiffness"][1]
                    twist_RVt[..., 2] = residual_action_wrench[..., 2] / control_para["translational_stiffness"][2]
                    twist_RVt[..., 3:6] = residual_action_wrench[..., 3:6] / control_para["rotational_stiffness"]
                    SE3_RVt = su.twc_to_SE3(twist_RVt)

                # # debug
                # # convert to tip position
                # SE3_Ttip = np.eye(4)
                # SE3_Ttip[0:3, 3] = np.array([0.0, 0.0, 0.297])
                # SE3_tipT = su.SE3_inv(SE3_Ttip)
                # # Tip0_T0 * T0_T1 * T1_tip1
                # for i in range(residual_action_horizon):
                #     SE3_BR_tip = SE3_tipT @ SE3_BR[i] @ SE3_Ttip
                #     # print("residual tip motion: ", SE3_BR_tip[:3, 3])
                #     if np.linalg.norm(SE3_BR_tip[:3, 3]) > 0.05:
                #         print("Warning: residual ref action is too large")
                #         print("rotation_mag: ", su.rotation_magnitude(SE3_BR[i][:3, :3]))


                #     SE3_Rvt_tip = SE3_tipT @ SE3_RVt[i] @ SE3_Ttip
                #     # print("vt tip motion: ", SE3_Rvt_tip[:3, 3])
                #     if np.linalg.norm(SE3_Rvt_tip[:3, 3]) > 0.05:
                #         print("Warning: residual vt action is too large")
                #         print("residual_action_wrench: ", residual_action_wrench[i])
                #         print("twist_RVt: ", twist_RVt[i])
                #         print("rotation_mag: ", su.rotation_magnitude(SE3_RVt[i][:3, :3]))



                # compute abs command on top of base policy action
                SE3_WR_all = [np.array] * len(id_list)
                SE3_WVt_all = [np.array] * len(id_list)
                gripper_base_all = [np.array] * len(id_list)

                for id in id_list:
                    # find the policy pose command closest to the residual action
                    # residual action: (obs_raw["robot_time_stamps_0"][-1]+residual_action_time_steps_s)*1000.0
                    # policy action: base_policy_action[f"policy_time_stamps_{id}"]
                    policy_indices = np.clip(np.searchsorted(
                        base_policy_action[f"policy_time_stamps_{id}"]/1000.0,
                        obs_raw["robot_time_stamps_0"][-1]+residual_action_time_steps_s, side='right'
                        )-1, 0, len(base_policy_action[f"policy_time_stamps_{id}"])-1)
                    SE3_WB = action_sparse_target_mats[id][policy_indices]     # [residual_action_horizon, 4, 4]
                    if residual_action_type == "pose9wrench6g1":
                        gripper_base = action_sparse_eoats[id][policy_indices]    # [residual_action_horizon, 1]


                    if residual_weights_loaded:
                        if control_para["scale_and_cap_residual_action"]:
                            SE3_WR = np.stack([
                                SE3_WB[i] @ scale_and_cap_residual_action(SE3_BR[i], scale=control_para["residual_action_scale_ratio"]) 
                                for i in range(residual_action_horizon)
                            ])    # [residual_action_horizon, 4, 4]
                            if "pose9wrench6" in residual_action_type:
                                SE3_WVt = np.stack([
                                    SE3_WR[i]
                                    @ scale_and_cap_residual_action(SE3_RVt[i], scale=control_para["residual_action_scale_ratio"])
                                    for i in range(residual_action_horizon)
                                ])      # residual_action_horizon, 4, 4]
                            else:
                                SE3_WVt = SE3_WR
                        else:
                            SE3_WR = np.stack([
                                SE3_WB[i] @ SE3_BR[i] for i in range(residual_action_horizon)
                            ])    # [residual_action_horizon, 4, 4]
                            if "pose9wrench6" in residual_action_type:
                                SE3_WVt = np.stack([
                                    SE3_WR[i] @ SE3_RVt[i]
                                    for i in range(residual_action_horizon)
                                ])      # residual_action_horizon, 4, 4]
                            else:
                                SE3_WVt = SE3_WR
                        if residual_action_type == "pose9wrench6g1":
                            gripper_residual = residual_action[..., 15:16]
                            gripper_final = gripper_residual + gripper_base
                    else:
                        SE3_WR = SE3_WB
                        SE3_WVt = SE3_WB
                        if residual_action_type == "pose9wrench6g1":
                            gripper_final = gripper_base

                    SE3_WR_all[id] = SE3_WR
                    SE3_WVt_all[id] = SE3_WVt


                outputs_ts_targets = [np.array] * len(id_list)
                outputs_gripper = [np.array] * len(id_list)
                for id in id_list:
                    # select output target: ref or virtual target
                    if control_para["test_nominal_target"]:
                        outputs_ts_targets[id] = su.SE3_to_pose7(SE3_WR_all[id])
                    else:
                        outputs_ts_targets[id] = su.SE3_to_pose7(SE3_WVt_all[id])
                    
                    if residual_action_type == "pose9wrench6g1":
                        outputs_gripper[id] = gripper_final


                # logging for plotting
                log_for_plot["residual_action_0_ref"].append(SE3_BR)
                if "pose9wrench6" in residual_action_type:
                    log_for_plot["residual_action_0_vt"].append(SE3_RVt)
                    log_for_plot["residual_action_wrench_0"].append(residual_action_wrench)
                log_for_plot["residual_action_0_abs"].append(outputs_ts_targets[0])
                log_for_plot["residual_action_time_stamps_0"].append(obs_raw["robot_time_stamps_0"][-1]+residual_action_time_steps_s)

                # the "now" when the observation is taken
                action_start_time_s = obs_raw["robot_time_stamps_0"][-1]
                timestamps = sparse_action_timesteps_s

                if len(id_list) == 1:
                    outputs_ts_targets = outputs_ts_targets[0].T  # N x 7 to 7 x N
                    outputs_gripper = outputs_gripper[0]  # N x 1
                else:
                    outputs_ts_targets = np.hstack(
                        outputs_ts_targets
                    ).T  # 2 x N x 7 to 14 x N
                    outputs_gripper = np.hstack(outputs_gripper) # 2 x N x 1 to N x 2

                if residual_action_type != "pose9wrench6g1":
                    outputs_gripper = None

                # check timing
                for id in id_list:
                    dt_rgb = env.current_hardware_time_s - obs_raw[f"rgb_time_stamps_{id}"][-1]
                    dt_ts_pose = (
                        env.current_hardware_time_s - obs_raw[f"robot_time_stamps_{id}"][-1]
                    )
                    dt_wrench = (
                        env.current_hardware_time_s - obs_raw[f"wrench_time_stamps_{id}"][-1]
                    )
                    # print(
                    #     f"[Total latency] obs lagging for robot {id}: dt_rgb: {dt_rgb}, dt_ts_pose: {dt_ts_pose}, dt_wrench: {dt_wrench}"
                    # )
                    if dt_rgb > control_para["delay_tolerance_s"] or dt_ts_pose > control_para["delay_tolerance_s"] or dt_wrench > control_para["delay_tolerance_s"]:
                        print("-------------------------------------------------------")
                        print("Delay larger than tolerance. Terminating episode.")
                        print("-------------------------------------------------------")
                        flag_large_delay = True
                        break

                if flag_large_delay:
                    break

                # send the action to the environment
                env.schedule_controls(
                    pose7_cmd=outputs_ts_targets,
                    eoat_cmd=outputs_gripper,
                    stiffness_matrices_6x6=outputs_ts_stiffnesses,
                    timestamps=(residual_action_time_steps_s + action_start_time_s) * 1000,
                )

            if flag_large_delay:
                break

            horizon_count += 1
            time_s = env.current_hardware_time_s
            sleep_duration_s = horizon_initial_time_s + execution_duration_s - time_s

            print("sleep_duration_s: ", sleep_duration_s)
            time.sleep(max(0, sleep_duration_s))

            if not control_para["pausing_mode"]:
                # only check duration when not in pausing mode
                if time_s - episode_initial_time_s > control_para["max_duration_s"]:
                    break

        print("End of episode.")
        # Online learning: send data to learner
        env.stop_saving_data()
        env.set_high_level_maintain_position()

        episode_folder = env.get_episode_folder()
        # save policy inferenced actions in zarr
        policy_inference_group = zarr.group(store=zarr.DirectoryStore(episode_folder + "/policy_inference.zarr"), overwrite=True)
        for id in id_list:
            policy_inference_group.create_dataset(f"ts_targets_{id}", data=np.array(base_action_targets_all[id]))
            if action_type == "pose9g1":
                policy_inference_group.create_dataset(f"ts_grippers_{id}", data=np.array(base_action_grippers_all[id]))

        # TODO: this is the only timestamp field on file that is in seconds, instead of milliseconds.
        # need to change this in the future
        policy_inference_group.create_dataset("timestamps_s", data=np.array(base_action_timestamps_all)/1000.0)

        ## Plot the episode
        if control_para["no_visual_mode"] == False:
            log_for_plot["rgb_0"] = log_for_plot["rgb_0"] 
            log_for_plot["rgb_time_stamps_0"] = np.array(log_for_plot["rgb_time_stamps_0"]) 
            log_for_plot["ts_pose_fb_0"] = np.array(log_for_plot["ts_pose_fb_0"]) 
            log_for_plot["robot_time_stamps_0"] = np.array(log_for_plot["robot_time_stamps_0"]) 
            log_for_plot["base_action_0"] = np.array(log_for_plot["base_action_0"]).reshape([-1, 7])
            log_for_plot["base_action_time_stamps_0"] = np.array(log_for_plot["base_action_time_stamps_0"]).reshape([-1])
            log_for_plot["residual_action_0_ref"] = np.array(log_for_plot["residual_action_0_ref"]).reshape([-1, 4, 4])
            if "pose9wrench6" in residual_action_type:
                log_for_plot["residual_action_0_vt"] = np.array(log_for_plot["residual_action_0_vt"]).reshape([-1, 4, 4])
                log_for_plot["residual_action_wrench_0"] = np.array(log_for_plot["residual_action_wrench_0"]).reshape([-1, 6])
            log_for_plot["residual_action_0_abs"] = np.array(log_for_plot["residual_action_0_abs"]).reshape([-1, 7])
            log_for_plot["residual_action_time_stamps_0"] = np.array(log_for_plot["residual_action_time_stamps_0"]).reshape([-1])

            # convert to tip position
            SE3_Ttip = np.eye(4)
            SE3_Ttip[0:3, 3] = np.array([0.0, 0.0, 0.297])
            # SE3_tipT = su.SE3_inv(SE3_Ttip)

            SE3_fb_WT = su.pose7_to_SE3(log_for_plot["ts_pose_fb_0"])
            SE3_base_policy_WT = su.pose7_to_SE3(log_for_plot["base_action_0"])
            # SE3_residual_Tref = log_for_plot["residual_action_0_ref"]
            # SE3_residual_Tvt = log_for_plot["residual_action_0_vt"]
            SE3_residual_WT = su.pose7_to_SE3(log_for_plot["residual_action_0_abs"])

            # residual = T0_T1
            # output: Tip0_tip1
            # Tip0_T0 * T0_T1 * T1_tip1

            SE3_fb_Wtip = SE3_fb_WT @ SE3_Ttip
            SE3_base_policy_Wtip = SE3_base_policy_WT @ SE3_Ttip
            SE3_residual_Wtip = SE3_residual_WT @ SE3_Ttip
            # SE3_residual_Tref_tip = SE3_tipT* SE3_residual_Tref @ SE3_Ttip
            # SE3_residual_Tvt_tip = SE3_tipT* SE3_residual_Tvt @ SE3_Ttip

            ts_tip_fb_0 = su.SE3_to_pose7(SE3_fb_Wtip)
            ts_tip_base_policy_0 = su.SE3_to_pose7(SE3_base_policy_Wtip)
            ts_tip_residual_0 = su.SE3_to_pose7(SE3_residual_Wtip)
            # ts_tip_residual_Tref_0 = su.SE3_to_pose7(SE3_residual_Tref_tip)
            # ts_tip_residual_Tvt_0 = su.SE3_to_pose7(SE3_residual_Tvt_tip)

            data_for_plot = {
                "ts_pose_fb_0": ts_tip_fb_0,
                "base_action_0": ts_tip_base_policy_0,
                "residual_action_0_abs": ts_tip_residual_0,
            }
            time_stamps_for_plot = {
                "ts_pose_fb_0": log_for_plot["robot_time_stamps_0"],
                "base_action_0": log_for_plot["base_action_time_stamps_0"],
                "residual_action_0_abs": log_for_plot["residual_action_time_stamps_0"],
            }
            if "pose9wrench6" in residual_action_type:
                additional_data_for_plot = {
                    "residual_action_wrench_0": log_for_plot["residual_action_wrench_0"]
                    # "residual_action_0_ref": log_for_plot["residual_action_0_ref"][..., 0:3, 3],
                    # "residual_action_0_vt": log_for_plot["residual_action_0_vt"][..., 0:3, 3],
                    # "ts_tip_residual_Tref_0": ts_tip_residual_Tref_0,
                    # "ts_tip_residual_Tvt_0": ts_tip_residual_Tvt_0,
                }
                additional_data_time_stamps = {
                    "residual_action_wrench_0": log_for_plot["residual_action_time_stamps_0"],
                    # "residual_action_0_ref": log_for_plot["residual_action_time_stamps_0"],
                    # "residual_action_0_vt": log_for_plot["residual_action_time_stamps_0"],
                }
                draw_timed_traj_and_rgb(rgb_images = log_for_plot["rgb_0"],
                                    rgb_time_stamps = log_for_plot["rgb_time_stamps_0"],
                                    data = data_for_plot,
                                    data_time_stamps = time_stamps_for_plot,
                                    additional_data = additional_data_for_plot,
                                    additional_data_time_stamps = additional_data_time_stamps,
                                    elev=20, azim=280)
            else:
                draw_timed_traj_and_rgb(rgb_images = log_for_plot["rgb_0"],
                                    rgb_time_stamps = log_for_plot["rgb_time_stamps_0"],
                                    data = data_for_plot,
                                    data_time_stamps = time_stamps_for_plot,
                                    elev=20, azim=280)

        if flag_large_delay:
            print("Large delay detected. deleting the episode.")
            if os.path.exists(episode_folder):
                shutil.rmtree(episode_folder)
                print("Found and deleted the new episode.")
            else:
                print("No new episode found! Something is wrong about path")
                exit(-1)
        else:
            print("Saved raw data for new episode: ", episode_folder)
            print("Do you want to keep it?")
            print("    d: delete the new episode.")
            print("    others: keep it and send it to learner.")
            c = input("Please select an option: ")
            if c == "d":
                print("Deleting the new episode.")
                if os.path.exists(episode_folder):
                    shutil.rmtree(episode_folder)
                    print("Found and deleted the new episode.")
                else:
                    print("No new episode found! Something is wrong about path")
                    exit(-1)
            else:
                if control_para["send_transitions_to_server"]:
                    print("Sending the new episode to learner.")
                    online_data = {
                        "episode_name": episode_folder[-18:], # only keep the episode name, such as episode_1742230408
                    }
                    actor_node.send_transitions(online_data)
                else:
                    print("Doing evaluation! Do not send data to learner.")
                num_episodes += 1

        print_env_status(env)

        print("Options:")
        print("     c: continue to next episode.")
        print("     j: reset, jog, calibrate, then continue.")
        print("     r: reset to default pose, then continue.")
        print("     b: reset to default pose, then quit the program.")
        print("     others: quit the program.")
        c = input("Please select an option: ")
        if c == "r" or c == "b" or c == "j":
            print("Resetting to default pose.")
            obs_raw = env.get_observation_from_buffer()
            N = 100
            duration_s = 5
            sample_indices = np.linspace(0, 1, N)
            timestamps = sample_indices * duration_s
            homing_ts_targets = np.zeros([7 * len(id_list), N])
            for id in id_list:
                ts_pose_fb = obs_raw[f"ts_pose_fb_{id}"][-1]
                
                pose7_waypoints = su.pose7_interp(ts_pose_fb, ts_pose_initial[id], sample_indices)
                homing_ts_targets[0 + id * 7 : 7 + id * 7, :] = pose7_waypoints.T

            time_now_s = env.current_hardware_time_s
            env.schedule_controls(
                pose7_cmd=homing_ts_targets,
                timestamps=(timestamps + time_now_s) * 1000,
            )
        elif c == "c":
            pass
        else:
            print("Quitting the program.")
            break

        if c == "b":
            input("Press Enter to qTrueuit program.")
            break

        if c == "j":
            input("Once robot is stopped, leave the robot free, Press Enter to run calibration.")
            print("---- Calibrating the robot. ----")
            env.calibrate_robot_wrench(NSamples = 100)
            print("---- Calibration done. ----")
            print_env_status(env)
            input("Hold the handle, Press Enter to enter a 3 second jog mode.")
            env.set_high_level_free_jogging()
            time.sleep(3)
            env.set_high_level_maintain_position()
            input("Jogging is done. Press Enter to continue.")

        print("Continuing to execution.")

    env.cleanup()


if __name__ == "__main__":
    main()
