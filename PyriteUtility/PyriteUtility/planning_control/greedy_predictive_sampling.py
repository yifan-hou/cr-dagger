import numpy as np
import pickle
import time

from PyriteUtility.planning_control.tree import Tree
from PyriteUtility.spatial_math import spatial_utilities as su

from PyriteGenesis.envs.genesis_base_task_env import State, Action

class GreedyPredictiveSampling():
    """
    Base class for predictive sampling
    
    """
    def __init__(self, config, batch_simulate_func, validify_action_func, log_folder_path):
        self.config = config
        self.batch_simulate_func = batch_simulate_func
        self.validify_action_func = validify_action_func
        self.log_folder_path = log_folder_path

    def rand_vel_sample(self, nominal_vel):
        N = self.config["n_samples"]
        dim = self.config["action_dim"]

        assert(len(nominal_vel.shape) == 1 and nominal_vel.shape[0] == dim), "nominal_vel must be a 1D array of length action_dim"
        if nominal_vel is None:
            nominal_vel = np.zeros(dim)

        robot_nominal_vel = nominal_vel[:-4] # Exclude the gripper joint
        gripper_vel = 0
        # Sample from a Gaussian distribution
        gripper_delta_action = np.random.normal(loc=robot_nominal_vel, scale=0.1, size=(N, 1, dim-4))

        # normalize the delta_action
        norm = np.linalg.norm(gripper_delta_action, axis=2, keepdims=True)
        norm[norm == 0] = 1  # avoid division by zero
        gripper_delta_action = gripper_delta_action / norm * self.config["action_mag_per_step"]

        return np.concatenate([gripper_delta_action, gripper_vel * np.ones((N, 1, 4))], axis=2)

    def update_action(self, action_samples, costs):
        # pick the action corresponding to the minimum cost
        min_cost_index = np.argmin(costs)
        action = action_samples[min_cost_index]
        return action, min_cost_index

    def solve(self,
                pose7_items, # (N_item, 7)
                robot_state,
                is_grasp,
                target_item_id, # used for validity check and grasping simulation
                pose7_WGoal_all):
        # randomly sample actions
        action_dim = self.config["action_dim"]
        action_horizon = self.config["action_horizon"]

        iter = 1
        action_all = []
        data_log = []

        pose7_items_current = pose7_items.copy()
        robot_state_current = robot_state.copy()
        robot_forces_current = np.zeros_like(robot_state_current)

        pose7_items_all = [pose7_items_current]
        robot_state_all = [robot_state_current]
        robot_forces_all = [robot_forces_current]

        action_selected = np.zeros((1, action_dim)) # (1, d)

        debug_action_sample_record = []
        debug_costs_record = []
        debug_aid_record = []

        pose7_item_initial = pose7_items_current[target_item_id, :].copy()  # Initial pose of the target item

        # compute inital cost with zero action
        action_sample = np.zeros((self.config["n_samples"], 1, action_dim))  # (n, h, d)
        states, actions, costs = self.batch_simulate_func(
            pose7_items=pose7_items_current,
            robot_state=robot_state_current,
            batch_actions=action_sample,
            is_grasp=is_grasp,
            target_item_id=target_item_id,
            pose7_WGoal_all=pose7_WGoal_all
        )
        cost_prev = costs["total"][0]
        cost_initial = cost_prev

        is_valid = False
        while iter <= action_horizon:
            # sample action perturbations (n, h, d)
            action_sample = self.rand_vel_sample(np.squeeze(action_selected)) # (n, 1, d)

            states, actions, costs = self.batch_simulate_func(
                pose7_items=pose7_items_current,
                robot_state=robot_state_current,
                batch_actions=action_sample,
                is_grasp = is_grasp,
                # batch_actions=batch_actions.reshape(-1, 2, action_dim),
                target_item_id=target_item_id,
                pose7_WGoal_all=pose7_WGoal_all
            )

            # NaN should get a high cost
            if states.hasNaN():
                # warning message
                print(f"[Predictive sampling] NaN detected in states at iter {iter}.")

            # update the ref action based on cost
            cost = costs["total"]
            action_selected, aid = self.update_action(action_sample, cost)
            cost_selected = cost[aid]

            # if the cost is not improved, break
            if cost_selected >= cost_prev - self.config["cost_improvement_threshold"]:
                print(f"[Predictive sampling] Cost did not improve enough at iter {iter}: {cost_prev} -> {cost_selected}. Breaking the loop.")
                break
            else:
                cost_prev = cost_selected

            action_all.append(np.squeeze(action_selected))
            action_selected = action_selected.reshape(1, -1)  # reshape to (1, d)

            # update the current pose7_items and robot_state
            pose7_items_current = states.pose7_items[aid, 1, ...].copy()
            robot_state_current = states.robot_dof_positions[aid, 1, ...].copy()
            robot_forces_current = states.robot_dof_forces[aid, 1, ...].copy()

            # Check validity here since we need the collision pairs
            is_valid = self.validify_action_func(
                pose7_item_initial=pose7_item_initial,
                pose7_item_final=pose7_items_current[target_item_id, :].copy(),
                target_item_id=target_item_id,
                pose7_WGoal_all=pose7_WGoal_all,
                num_envs_used=action_sample.shape[0],
                env_idx=aid,
            )

            # logs
            if self.log_folder_path is not None:
                debug_action_sample_record.append(action_sample)
                debug_costs_record.append(cost)
                debug_aid_record.append(aid)

                pose7_items_all.append(pose7_items_current)
                robot_state_all.append(robot_state_current)
                robot_forces_all.append(robot_forces_current)

            print(f"[Predictive sampling] Iter {iter}/{self.config['action_horizon']}, Cost mean: {np.mean(cost)}, best cost: {np.min(cost)}")
            iter += 1

        if cost_selected > cost_initial - 5*self.config["cost_improvement_threshold"]:
            is_valid = False

        # log the data
        states_all = State()
        states_all.pose7_items = np.array(pose7_items_all)[np.newaxis, ...] # add env axis
        states_all.robot_dof_positions = np.array(robot_state_all)[np.newaxis, ...] # add env axis
        states_all.robot_dof_forces = np.array(robot_forces_all)[np.newaxis, ...] # add env axis
        action_all = np.array(action_all)
        if self.log_folder_path is not None:
            data_log.append({
                "iter": 0,
                "states": states_all,
                "actions": action_all,
                "costs": costs,
                "is_valid": is_valid,
                "timestamp": time.time(),
                "debug_action_sample_record": debug_action_sample_record,
                "debug_costs_record": debug_costs_record,
                "debug_aid_record": debug_aid_record,
            })
        return states_all, action_all, cost, is_valid, data_log



if __name__ == "__main__":
    # Example usage
    config = {
        "action_horizon": 20,
        "action_dim": 7,
        "n_samples": 2 #2048,
    }
    
    predictive_sampling = GreedyPredictiveSampling(config)

    sample = predictive_sampling.rand_vel_sample()
    print("Sampled actions:", sample)
