import numpy as np
import time
import pickle

from PyriteUtility.planning_control.tree import Tree
from PyriteUtility.spatial_math import spatial_utilities as su

class RRT(Tree):
    """
    Rapidly-exploring Random Tree (RRT) class for path planning.
    
    Inherits from the Tree class and implements the RRT algorithm.
    """
    
    def __init__(self,
                 rrt_config,
                 number_of_items,
                 initial_node_properties,
                 goal_items_pose7_overwrite, # overrite the goal in rrt_config
                 env_config_name,
                 steer_solver_func,
                 compute_feasible_hand_configs_func):
        """
        Initialize the RRT with a start and goal position.
        """
        # scene
        node_property_schema = {
            "pose7_items": np.ndarray,  # Assuming position is a numpy array
            "pose7_items_stabilized": np.ndarray,
            # the following properties are consistent with the incoming edge
            "target_item_id": int,  # ID of the item being manipulated
            "joints_current_hand": np.ndarray,  # Hand joint positions (n,)
            "is_grasp": bool,  # Whether the hand is in a grasp pose
        }
        edge_property_schema = {
            "pose7_items": np.ndarray,
            "robot_dof_positions": np.ndarray,
            "robot_dof_forces": np.ndarray,
        }

        # initialize the tree
        super().__init__(node_property_schema = node_property_schema, 
                 edge_property_schema = edge_property_schema, 
                 initial_capacity = rrt_config["tree_initial_capacity"], growth_factor = rrt_config["tree_growth_factor"])

        # initialize the first node of the tree
        self.add_node(node_id = 0,
                      properties = initial_node_properties)
 
        self.env_config_name = env_config_name
        self.number_of_items = number_of_items
        
        self.max_iters = rrt_config["max_iterations"]
        self.rot_weight = rrt_config["rot_weight"]
        self.max_num_steer_samples = rrt_config["max_num_steer_samples"]
        pose3_range = np.array(rrt_config["pose3_range"])
        self.items_pose3_lower = np.tile(pose3_range[:, 0].T, (self.number_of_items, 1))
        self.items_pose3_upper = np.tile(pose3_range[:, 1].T, (self.number_of_items, 1))
        self.max_new_nodes = rrt_config["max_new_nodes"]
        self.goal_sampling_probability =  rrt_config["goal_sampling_probability"]
        self.save_steering_log = rrt_config["save_steering_log"]

        self.goal_items_pose7 = goal_items_pose7_overwrite
        self.goal_distance_threshold = rrt_config["goal_distance_threshold"]

        self.steer_solver_func = steer_solver_func
        self.compute_feasible_hand_configs_func = compute_feasible_hand_configs_func

    def sample_random_states(self):
        """
        Sample a random pose for each object.
        
        Returns:
            np.ndarray: A (n x 7) np array
        """
        # sample a random number between 0 and 1
        random_number = np.random.rand()
        if  random_number < self.goal_sampling_probability:
            # Sample the goal item
            pose7_all = self.goal_items_pose7
        else:
            # Sample a random pose within the defined range
            pose3_all = np.random.uniform(self.items_pose3_lower, self.items_pose3_upper, size=(self.number_of_items, 3))
            # Sample a random quaternion
            quaternion = np.random.rand(self.number_of_items, 4)
            quaternion /= np.linalg.norm(quaternion, axis=1, keepdims=True)  # Normalize to get a valid quaternion
            # Combine the position and quaternion into a 7D vector
            pose7_all = np.concatenate((pose3_all, quaternion), axis=1)

        return pose7_all # (n, 7)

    def find_nearest_node(self, target_state):
        """
        Find the nearest node in the tree to the target item pose.
        
        Args:
            target_state: (n_items, 7) The pose of all items.
        
        Returns:
            int: The ID of the nearest node in the tree.
        """
        pose7_all_nodes_all_items = self.get_node_property_concatenated('pose7_items') # (N_nodes, n_items, 7)

        # Calculate distances from all nodes to the target pose
        distances_all = su.dist_pose7(pose7_all_nodes_all_items, target_state, rot_weight=self.rot_weight) # (N_nodes, n_items)
        distances_average = np.mean(distances_all, axis=1)  # Average distance across all items
        nearest_node_index = np.argmin(distances_average)  # Index of the nearest node

        # print(f"[RRT.find_nearest_node] Nearest node index: {nearest_node_index}, Distance: {distances_average[nearest_node_index]}")
        # input("Press Enter to continue...")

        return self._index_to_node_id[nearest_node_index]


    def steer(self, nearest_node_id, pose7_WGoal_all):
        """
        Steer the RRT tree towards the goal pose for the target item.

        Solve steer twice:
            1. First time just for the target item under the current hand pose from the current state
            2. Second time, solve for all items from the stabilized states, sample all possible hand configs. Do not include the target item if step 1 already found solutions.

        Args:
            nearest_node_id: The ID of the nearest node in the tree.
            pose7_WGoal_all: The goal pose for ALL the items.
        """
        nearest_node = self.nodes[nearest_node_id]
        item_poses_current = nearest_node.properties['pose7_items'] # (n_items, 7)
        item_poses_stabilized = nearest_node.properties['pose7_items_stabilized'] # (n_items, 7)
        target_item_id_current = nearest_node.properties['target_item_id']
        hand_pose_current = nearest_node.properties['joints_current_hand'] # (n,)
        is_grasp_current = nearest_node.properties['is_grasp']

        # determine which items have not yet reached their goal
        item_not_at_goal_ids = []
        for i in range(self.number_of_items):
            dist_to_goal = su.dist_pose7(nearest_node.properties['pose7_items'][i], pose7_WGoal_all[i], rot_weight=self.rot_weight)
            if dist_to_goal > self.goal_distance_threshold:
                item_not_at_goal_ids.append(i)
        
        first_states_list = []
        first_actions_list = []
        first_pose7_items_stabilized = []
        first_target_item_ids = []

        if target_item_id_current in item_not_at_goal_ids:
            # First steering attempt: try to move the target item under the current hand configuration
            if is_grasp_current is not None: # if None, it means the first node, no grasp assigned to this node
                first_states_list, first_actions_list, first_pose7_items_stabilized, first_target_item_ids = self.steer_solver_func(
                    pose7_items = item_poses_current,
                    robot_state = np.array([hand_pose_current]),
                    is_grasp = np.array([is_grasp_current]),
                    target_item_id = [target_item_id_current],
                    pose7_WGoal_all = pose7_WGoal_all,
                )

            if len(first_states_list) > 0:
                # successfully steered the target item using the current hand configuration. No need to try its stabilized pose
                item_not_at_goal_ids.remove(target_item_id_current)
        
        # Second steering attempt: consider all items not yet reached their goal, resample hand configurations
        hand_poses_all = []
        is_grasp_pose_all = []
        target_item_ids_all = []
        total_feasible_poses = 0
        for target_item_id in item_not_at_goal_ids:
            feasible_grasp_poses, feasible_probe_poses = self.compute_feasible_hand_configs_func(nearest_node.properties['pose7_items'], target_item_id)

            num_feasible_grasp_poses = feasible_grasp_poses.shape[0]
            num_feasible_probe_poses = feasible_probe_poses.shape[0]

            if num_feasible_grasp_poses + num_feasible_probe_poses > self.max_num_steer_samples:
                # try to sample equal amount of grasp poses and probe poses
                if num_feasible_grasp_poses > num_feasible_probe_poses:
                    num_grasp_to_sample = min(num_feasible_grasp_poses, self.max_num_steer_samples // 2)
                    num_probe_to_sample = self.max_num_steer_samples - num_grasp_to_sample
                    if num_probe_to_sample > num_feasible_probe_poses:
                        num_probe_to_sample = num_feasible_probe_poses
                        num_grasp_to_sample = self.max_num_steer_samples - num_probe_to_sample
                else:
                    num_probe_to_sample = min(num_feasible_probe_poses, self.max_num_steer_samples // 2)
                    num_grasp_to_sample = self.max_num_steer_samples - num_probe_to_sample
                    if num_grasp_to_sample > num_feasible_grasp_poses:
                        num_grasp_to_sample = num_feasible_grasp_poses
                        num_probe_to_sample = self.max_num_steer_samples - num_grasp_to_sample

                # Randomly sample a subset of feasible hand poses
                feasible_grasp_poses = feasible_grasp_poses[np.random.choice(feasible_grasp_poses.shape[0], num_grasp_to_sample, replace=False)]
                feasible_probe_poses = feasible_probe_poses[np.random.choice(feasible_probe_poses.shape[0], num_probe_to_sample, replace=False)]
            
            if num_feasible_grasp_poses == 0:
                feasible_hand_poses = feasible_probe_poses
                is_grasp_pose = np.zeros(feasible_probe_poses.shape[0])
            elif num_feasible_probe_poses == 0:
                feasible_hand_poses = feasible_grasp_poses
                is_grasp_pose = np.ones(feasible_grasp_poses.shape[0])
            else:
                # concatenate the feasible grasp and probe poses
                feasible_hand_poses = np.concatenate((feasible_grasp_poses, feasible_probe_poses), axis=0)
                # create a vector to indicate whether a hand pose is a grasp or not
                is_grasp_pose = np.concatenate((np.ones(feasible_grasp_poses.shape[0]), np.zeros(feasible_probe_poses.shape[0])), axis=0)

            if feasible_hand_poses.shape[0] == 0:
                continue
            total_feasible_poses += feasible_hand_poses.shape[0]
            hand_poses_all.append(feasible_hand_poses)
            is_grasp_pose_all.append(is_grasp_pose)
            target_item_ids_all += [target_item_id] * feasible_hand_poses.shape[0]

        if total_feasible_poses == 0:
            # Worst possible case -_-
            print(f"[RRT.steer] No feasible hand configurations found at node {nearest_node_id}.")
            return first_states_list, first_actions_list, first_pose7_items_stabilized, first_target_item_ids

        hand_poses_all = np.concatenate(hand_poses_all, axis=0)
        is_grasp_pose_all = np.concatenate(is_grasp_pose_all, axis=0)

        # Iterate through all feasible hand poses
        print(f"[RRT.steer] Trying {hand_poses_all.shape[0]} feasible hand poses for item {target_item_ids_all} at node {nearest_node_id}.")
        second_states_list, second_actions_list, second_pose7_items_stabilized, second_target_item_ids = self.steer_solver_func(
            pose7_items = item_poses_stabilized,
            robot_state = hand_poses_all,
            is_grasp = is_grasp_pose_all,
            target_item_id = target_item_ids_all,
            pose7_WGoal_all = pose7_WGoal_all,
        )

        # Combine the results from both steering attempts
        all_states_list = first_states_list + second_states_list
        all_actions_list = first_actions_list + second_actions_list
        all_pose7_items_stabilized = first_pose7_items_stabilized + second_pose7_items_stabilized
        all_target_item_ids = first_target_item_ids + second_target_item_ids
        return all_states_list, all_actions_list, all_pose7_items_stabilized, all_target_item_ids



    def plan(self, env, log_folder_path=None):
        """
        Run the RRT algorithm to grow the existing tree towards the goal.
        
        Returns:
            leaf_id (int): The ID of the leaf node that reached the goal, or None if no path was found.
        """
        if not self._arrays_initialized:
            raise RuntimeError("[RRT] Tree not initialized. Call set_initial_node() first.")

        new_nodes = 0
        n_iters = 0
        while True:
            n_iters += 1
            if n_iters > self.max_iters:
                print(f"[RRT] Max iterations reached without finding a path.")
                break

            print(f"[RRT] Iteration {n_iters}, New Nodes: {new_nodes}/{self.max_new_nodes}")
            # Sample a random point in the space
            sampled_goal_item_poses = self.sample_random_states()

            # Find the nearest node in the tree to the random point
            nearest_node_id = self.find_nearest_node(sampled_goal_item_poses)

            # find the target item as the one that is furthest away from its goal
            current_item_poses = self.nodes[nearest_node_id].properties['pose7_items'] # (n_items, 7)
            distances_to_goal = su.dist_pose7(sampled_goal_item_poses, current_item_poses, rot_weight=self.rot_weight) # (n_items,)
            target_item_id = np.argmax(distances_to_goal)

            # Steer from the nearest node towards the random point
            result_states, result_actions, result_stabilized_item_poses, result_target_item_ids = self.steer(nearest_node_id, sampled_goal_item_poses)
            if len(result_states) == 0:
                print(f"[RRT] No valid steering found for target item {target_item_id}.")
                continue
            print(f"[RRT] Steering solved for node {nearest_node_id}, item {target_item_id}, found {len(result_states)} feasible states.")

            for state, action, pose7_items_stabilized, target_item_id in zip(result_states, result_actions, result_stabilized_item_poses, result_target_item_ids):
                # Create a new node with the properties of the state
                assert len(state.is_grasp) == 1, "Sanity check: Here each state should correspond to a single environment."
                new_node_properties = {
                    "pose7_items": np.array(state.pose7_items[-1, :, :]),  # Last frame
                    "pose7_items_stabilized": pose7_items_stabilized,  # Stabilized pose after the action
                    "target_item_id": target_item_id,
                    "joints_current_hand": np.array(state.robot_dof_positions[-1, :]),  # Last frame
                    "is_grasp": bool(state.is_grasp[0]),
                }

                new_edge_properties = {
                    "pose7_items": np.array(state.pose7_items),  # Whole traj
                    "robot_dof_positions": np.array(state.robot_dof_positions),  # Whole traj
                    "robot_dof_forces": np.array(state.robot_dof_forces),  # Whole traj
                }

                # Add the new node to the tree
                new_node_id = self.node_count
                self.add_node(node_id=new_node_id,
                            properties=new_node_properties,
                            parent_id=nearest_node_id)

                if self.save_steering_log and log_folder_path is not None:
                    log_info = {
                        "env_config_name": self.env_config_name,
                        "new_node_id": new_node_id,
                        "nearest_node_id": nearest_node_id,
                        "target_item_id": target_item_id,  # ID of the goal item to manipulate
                        "sampled_goal_item_poses": sampled_goal_item_poses,
                        "state": state,
                        "action": action,
                    }
                    log_file_path = log_folder_path + f"/steer_{time.time()}.pkl"
                    with open(log_file_path, 'wb') as f:
                        pickle.dump(log_info, f)

                # Add an edge from the nearest node to the new node
                self.add_edge_properties(nearest_node_id, new_node_id, new_edge_properties)
                new_nodes += 1

                new_item_poses = state.pose7_items[-1, :, :]
                if np.mean(su.dist_pose7(new_item_poses, self.goal_items_pose7, self.rot_weight)) < self.goal_distance_threshold:
                    # The goal is reached! Return the path
                    print(f"[RRT] Goal reached at node {new_node_id}.")

                    reverse_path = self.get_path_to_root(new_node_id)
                    # reverse order of the path
                    rrt_path = reverse_path[::-1]
                    Nnodes = len(rrt_path)
                    edges = []
                    for n in range(Nnodes - 1):
                        start_id = rrt_path[n].id
                        end_id = rrt_path[n + 1].id
                        edge = self.get_edge(start_id, end_id)
                        edges.append(edge)
                        
                        # debug
                        edge_properties = edge.properties
                        pose7_items_all = edge_properties["pose7_items"]
                        robot_dof_forces = edge_properties["robot_dof_forces"]
                        if len(pose7_items_all) != len(robot_dof_forces):
                            print(f"[RRT.plan] Edge from {start_id} to {end_id} has inconsistent lengths: pose7_items {len(pose7_items_all)}, robot_dof_forces {len(robot_dof_forces)}")
                            assert False
                        
                    
                    return rrt_path, edges
                
                # # goal is not reached, then need to update the new node state to the stabilized state
                # self.update_node_property(new_node_id, "pose7_items", pose7_items_stabilized)

            # Check if we have reached the maximum number of new nodes
            if new_nodes >= self.max_new_nodes:
                print(f"[RRT] Reached maximum new nodes: {new_nodes}. Stopping.")
                break

        return None, None