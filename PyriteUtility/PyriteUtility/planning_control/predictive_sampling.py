import numpy as np
import pickle
import time

from PyriteUtility.planning_control.tree import Tree
from PyriteUtility.spatial_math import spatial_utilities as su

class PredictiveSampling():
    """
    Base class for predictive sampling
    
    """
    def __init__(self, config, batch_simulate_func, validify_action_func, log_folder_path):
        self.config = config
        self.batch_simulate_func = batch_simulate_func
        self.validify_action_func = validify_action_func
        self.log_folder_path = log_folder_path

    def noise_sample(self, iter):
        N = self.config["n_diffusion_steps"]
        H = self.config["action_horizon"]
        beta = self.config["dial_mpc"]["beta"]
        assert(iter > 0), "iter must be greater than 0 for noise sampling"
        # Dial-MPC
        cov_mag = np.exp(-iter/beta/N)
        cov = cov_mag * np.eye(self.config["action_dim"])
        # sample noise
        delta_action = np.random.multivariate_normal(
            mean=np.zeros(self.config["action_dim"]),
            cov=cov,
            size=(self.config["n_samples"], H)
        )
        # limit the action magnitude
        delta_action = np.clip(delta_action, self.config["action_bound_lower"], self.config["action_bound_upper"])
        # delta_action shape: (n, h, d)
        return delta_action

    def update_action(self, action, delta_actions, costs):
        lbd = self.config["dial_mpc"]["lambda"]
        exp_weights = np.reshape(np.exp(- costs / lbd), (-1, 1, 1))
        nominator = np.sum(exp_weights * delta_actions, axis=0)  # (h, d)
        denominator = np.sum(exp_weights) # scalar
        if (np.abs(denominator) < 1e-9).any():
            raise ValueError("Denominator is too small, cannot update action.")
        action = action + nominator / denominator  # (h, d)
        return action

    def solve(self,
                pose7_items, # (N_item, 7)
                robot_state,
                is_grasp,
                target_item_id,
                pose7_WGoal_all,
                log_folder_path=None):
        # randomly sample actions
        action_dim = self.config["action_dim"]
        action_horizon = self.config["action_horizon"]

        iter = 1
        action = np.zeros((action_horizon, action_dim)) # (h, d)
        data_log = []
        cost_prev = 1e9
        cost_initial = 0
        while iter <= self.config["n_diffusion_steps"]:
            # sample action disturbations (n, h, d)
            delta_action = self.noise_sample(iter)

            # update the action samples
            action_sample = action + delta_action

            # rollout the action samples, get costs
            states, actions, costs = self.batch_simulate_func(
                pose7_items=pose7_items,
                robot_state=robot_state,
                batch_actions=action_sample,
                is_grasp=is_grasp,
                target_item_id=target_item_id,
                pose7_WGoal_all=pose7_WGoal_all
            )
            # TODO: properly check validity
            is_valid = True

            cost = costs["total"]
            cost_mean = np.mean(cost)
            if iter == 1:
                cost_initial = cost_mean

            # if the cost is not improved, break
            if cost_mean >= cost_prev - self.config["cost_improvement_threshold"]:
                # print(f"[Predictive sampling] Cost did not improve enough at iter {iter}: {cost_prev} -> {cost_mean}. Breaking the loop.")
                break
            else:
                cost_prev = cost_mean


            # NaN should get a high cost
            if states.hasNaN():
                # warning message
                print(f"[Predictive sampling] NaN detected in states at iter {iter}.")

            # log the data
            if self.log_folder_path is not None:
                data_log.append({
                    "iter": iter,
                    "states": states,
                    "actions": actions,
                    "costs": costs,
                    "is_valid": is_valid,
                    "timestamp": time.time()
                })

            # update the ref action based on cost
            action = self.update_action(action, delta_action, cost)

            # logs
            print(f"[Predictive sampling] Iter {iter}/{self.config['n_diffusion_steps']}, Cost mean: {cost_mean}, best cost: {np.min(cost)}")

            iter += 1

        if cost_mean > cost_initial - self.config["cost_validity_threshold"]:
            is_valid = False
        return states, actions, cost, is_valid, data_log

if __name__ == "__main__":
    # Example usage
    config = {
        "n_diffusion_steps": 10,
        "action_horizon": 20,
        "action_dim": 7,
        "dial_mpc": {
            "beta": 0.02, # 0.01
            "lambda": 0.01
        },
        "n_samples": 1 #2048,
    }
    
    predictive_sampling = PredictiveSampling(config)
    
    for i in range(config["n_diffusion_steps"]):
        print(f"Noise sample for iteration {i}:")
        noise_sample = predictive_sampling.noise_sample(i+1)
        mag = np.linalg.norm(noise_sample, axis=(1, 2))
        max = np.max(np.abs(noise_sample))
        print(f"Max noise magnitude: {max}, Mean noise magnitude: {mag}")
    
    # # Define a dummy batch_simulate_func for testing
    # def dummy_batch_simulate_func(action_sample):
    #     # Simulate some states, actions, costs, and validity
    #     states = np.random.rand(2048, 20, 3, 7)  # Dummy states
    #     actions = np.random.rand(2048, 20, 7)   # Dummy actions
    #     cost = np.random.rand(2048)            # Dummy costs
    #     is_valid = True                           # Dummy validity
    #     return states, actions, cost, is_valid
    
    # states, action, cost, is_valid = predictive_sampling.solve(dummy_batch_simulate_func)
    # print("Final action:", action)