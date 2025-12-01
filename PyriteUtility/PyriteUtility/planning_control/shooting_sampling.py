import numpy as np
import pickle
import time

from PyriteUtility.planning_control.tree import Tree
from PyriteUtility.spatial_math import spatial_utilities as su

from PyriteGenesis.envs.genesis_base_task_env import State, Action

class ShootingSampling():
    """
    Base class for shooting sampling
    
    """
    def __init__(self, config, batch_simulate_func, validify_action_func, log_folder_path):
        self.config = config
        self.batch_simulate_func = batch_simulate_func
        self.log_folder_path = log_folder_path

    def rand_vel_sample(self, n_vel_samples):
        dim = self.config["action_dim"]

        nominal_vel = np.zeros(dim)

        robot_nominal_vel = nominal_vel[:-4] # Exclude the gripper joint
        gripper_vel = 0
        # Sample from a Gaussian distribution
        gripper_delta_action = np.random.normal(loc=robot_nominal_vel, scale=0.1, size=(n_vel_samples, 1, dim-4))

        # normalize the delta_action
        norm = np.linalg.norm(gripper_delta_action, axis=2, keepdims=True)
        norm[norm == 0] = 1  # avoid division by zero
        gripper_delta_action = gripper_delta_action / norm * self.config["action_mag_per_step"]

        return np.concatenate([gripper_delta_action, gripper_vel * np.ones((n_vel_samples, 1, 4))], axis=2)

    def update_action(self, action_samples, costs):
        # pick the action corresponding to the minimum cost
        min_cost_index = np.argmin(costs)
        action = action_samples[min_cost_index]
        return action, min_cost_index

    def solve(self,
                pose7_items, # (N_item, 7)
                robot_state, # (N, D)
                is_grasp, # (N,)
                target_item_id, # (N,)
                pose7_WGoal_all):

        N = len(robot_state)
        n_vel_samples = np.floor(self.config["num_envs"] / N).astype(int)

        action_sample = self.rand_vel_sample(n_vel_samples) # (ns, 1, d)
        states_all, actions_all, env_safety_num_frames, pose7_items_stabilized, target_item_ids_all = self.batch_simulate_func(
            pose7_items=pose7_items,
            robot_state=robot_state,
            batch_actions=action_sample,
            is_grasp = is_grasp,
            target_item_id=target_item_id,
            pose7_WGoal_all=pose7_WGoal_all
        )
        states_list = states_all.extract_to_list(env_safety_num_frames, 3)
        actions_list = actions_all.extract_to_list(env_safety_num_frames, 3)
        pose7_items_stabilized = list(pose7_items_stabilized[env_safety_num_frames >= 3])
        target_item_ids = list(target_item_ids_all[env_safety_num_frames >= 3])
        # data_log.append({
        #     "states": states_list,
        #     "actions": actions_list,
        # })
        return states_list, actions_list, pose7_items_stabilized, target_item_ids

if __name__ == "__main__":
    # Example usage
    config = {
        "action_horizon": 20,
        "action_dim": 7,
        "n_samples": 2 #2048,
    }
    
    predictive_sampling = ShootingSampling(config)

    sample = predictive_sampling.rand_vel_sample(None)
    print("Sampled actions:", sample)
