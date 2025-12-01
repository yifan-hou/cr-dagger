from unittest import result
import numpy as np

from PyriteUtility.spatial_math import spatial_utilities as su

class BoxGraspSampler():
    """
    A class to sample gripper grasp poses on a box. The box is centered
    at the origin.
    
    There are two types of grasp poses:
        probing: fingers are closed, only tip of the fingers are in contact with the box surface
        grasping: fingers are open, the gripper is grasping the box

    For probing, we precompute fingertip locations and grasp z axes.
    A pose is represented as (x, y, z, ax, ay, az, width), where:
        x, y, z: position of the fingertip center
        ax, ay, az: direction of the grasp z axis
        width: width of the gripper
    
    For grasping, a grasp pose is represented as (x, y, z, qx, qy, qz, qw, width).

    Args:
        box_size (np.ndarray): Size of the box in the form of [length_x, length_y, length_z].
        pose7_TipHand (np.ndarray): Pose of the gripper tip in the hand frame. We define the tip frame to be
                at the center of the gripper fingertips, with Y pointing from palm to fingertips,
                X pointing from left to right (aka the grasp axis). We first construct the tip frames, then
                use this argument to compute the hand frames. 
        config (dict): Configuration dictionary containing:
            - np_per_edge (list[int]): Number of probing points per edge [nx, ny, nz].
            - robot_tilting_tol_rad (float): Tolerance for the robot tilting in radians.
            - edge_buffer (float): Buffer distance from the edge for probing.
            - grasp_edge_buffer (float): Offset for the grasping pose from the box surface.
            - grasp_penetration (float): Finger penetration depth at grasping.

    """
    def __init__(self, box_size, pose7_TipHand, config):
        self.robot_tilting_tol_rad = config["robot_tilting_tol_deg"] * np.pi / 180.0 
        ##
        ## probing pose
        ##
        np_per_edge = config["np_per_edge"]
        lx = box_size[0]/2.0
        ly = box_size[1]/2.0
        lz = box_size[2]/2.0
        nx = np_per_edge[0]
        ny = np_per_edge[1]
        nz = np_per_edge[2]

        edge_buffer = config["edge_buffer"]

        # generate fingertip center positions for probing contact
        x_range = np.linspace(-lx+edge_buffer, lx-edge_buffer, nx)
        y_range = np.linspace(-ly+edge_buffer, ly-edge_buffer, ny)
        z_range = np.linspace(-lz+edge_buffer, lz-edge_buffer, nz)
        top_fingertips = np.array([[x, y, lz] for x in x_range for y in y_range])
        bottom_fingertips = np.array([[x, y, -lz] for x in x_range for y in y_range])
        left_fingertips = np.array([[x, -ly, z] for x in x_range for z in z_range])
        right_fingertips = np.array([[x, ly, z] for x in x_range for z in z_range])
        front_fingertips = np.array([[lx, y, z] for y in y_range for z in z_range])
        back_fingertips = np.array([[-lx, y, z] for y in y_range for z in z_range])
        
        altitude_degrees = config["probing_altitude_degrees"]
        longitude_num = config["probing_longitude_num"]

        altitude = np.array(altitude_degrees) * np.pi / 180.0  # 35 degrees and 70 degrees in radians
        longitude = np.linspace(0, 2 * np.pi, longitude_num, endpoint=False)  # 8 evenly spaced angles around the circle
        # generate hand axes (tip to palm) spanning in a fan shape
        hand_axes = [[np.sin(a)*np.cos(b), np.sin(a)*np.sin(b), np.cos(a)] for a in altitude for b in longitude]
        hand_axes = np.array(hand_axes)

        hand_top = hand_axes
        hand_bottom = np.column_stack((hand_axes[:, 0], hand_axes[:, 1], -hand_axes[:, 2]))
        hand_left = np.column_stack((hand_axes[:, 0], -hand_axes[:, 2], hand_axes[:, 1]))
        hand_right = np.column_stack((hand_axes[:, 0], hand_axes[:, 2], -hand_axes[:, 1]))
        hand_front = np.column_stack((hand_axes[:, 2], hand_axes[:, 1], -hand_axes[:, 0]))
        hand_back = np.column_stack((-hand_axes[:, 2], hand_axes[:, 1], hand_axes[:, 0]))

        # Assemble each combination of fingertip_positions and grasp_hand_axes into a (n, 6) matrix
        probing_poses_top = np.array([
            np.concatenate((fingertip, hand_axis)) for fingertip in top_fingertips for hand_axis in hand_top
        ])
        probing_poses_bottom = np.array([
            np.concatenate((fingertip, hand_axis)) for fingertip in bottom_fingertips for hand_axis in hand_bottom
        ])
        probing_poses_left = np.array([
            np.concatenate((fingertip, hand_axis)) for fingertip in left_fingertips for hand_axis in hand_left
        ])
        probing_poses_right = np.array([
            np.concatenate((fingertip, hand_axis)) for fingertip in right_fingertips for hand_axis in hand_right
        ])
        probing_poses_front = np.array([
            np.concatenate((fingertip, hand_axis)) for fingertip in front_fingertips for hand_axis in hand_front
        ])
        probing_poses_back = np.array([
            np.concatenate((fingertip, hand_axis)) for fingertip in back_fingertips for hand_axis in hand_back
        ])
        # Concatenate all probing poses
        self.tip_probing_poses = np.concatenate([
            probing_poses_top, probing_poses_bottom,
            probing_poses_left, probing_poses_right,
            probing_poses_front, probing_poses_back
        ], axis=0)
        # Add width to probing poses
        self.tip_probing_poses = np.concatenate(
            (self.tip_probing_poses, np.full((self.tip_probing_poses.shape[0], 1), 0.0)), axis=1
        )  # width is set to 0 for probing poses
        ##
        ## grasping pose
        ##
        grasp_side_edge_buffer = config["grasp_side_edge_buffer"]
        grasp_depth = config["grasp_depth"]
        # grasp_edge_buffer = 0.05
        x_range = np.linspace(-lx+grasp_side_edge_buffer, lx-grasp_side_edge_buffer, nx)
        y_range = np.linspace(-ly+grasp_side_edge_buffer, ly-grasp_side_edge_buffer, ny)
        z_range = np.linspace(-lz+grasp_side_edge_buffer, lz-grasp_side_edge_buffer, nz)
        top_fingertips1 = np.array([[x, 0, lz-grasp_depth] for x in x_range])
        top_fingertips2 = np.array([[0, y, lz-grasp_depth] for y in y_range])
        bottom_fingertips1 = np.array([[x, 0, -lz+grasp_depth] for x in x_range])
        bottom_fingertips2 = np.array([[0, y, -lz+grasp_depth] for y in y_range])
        left_fingertips1 = np.array([[x, -ly+grasp_depth, 0] for x in x_range])
        left_fingertips2 = np.array([[0, -ly+grasp_depth, z] for z in z_range])
        right_fingertips1 = np.array([[x, ly-grasp_depth, 0] for x in x_range])
        right_fingertips2 = np.array([[0, ly-grasp_depth, z] for z in z_range])
        front_fingertips1 = np.array([[lx-grasp_depth, y, 0] for y in y_range])
        front_fingertips2 = np.array([[lx-grasp_depth, 0, z] for z in z_range])
        back_fingertips1 = np.array([[-lx+grasp_depth, y, 0] for y in y_range])
        back_fingertips2 = np.array([[-lx+grasp_depth, 0, z] for z in z_range])

        R90x = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
        R90y = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
        R90z = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        R180x = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
        R180y = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])
        R180z = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])

        top1_quat = su.SO3_to_quat(R90x.T @ R90y.T)
        top2_quat = su.SO3_to_quat(R90x.T)
        bottom1_quat = su.SO3_to_quat(R90x @ R90y)
        bottom2_quat = su.SO3_to_quat(R90x)
        left1_quat = su.SO3_to_quat(R90y)
        left2_quat = np.array([1, 0, 0, 0])
        right1_quat = su.SO3_to_quat(R180z @ R90y.T)
        right2_quat = su.SO3_to_quat(R180z)
        front1_quat = su.SO3_to_quat(R90z @ R90y)
        front2_quat = su.SO3_to_quat(R90z)
        back1_quat = su.SO3_to_quat(R90z.T @ R90y)
        back2_quat = su.SO3_to_quat(R90z.T)
        
        grasp_penetration = config["grasp_penetration"]
        top1_width = box_size[1] - grasp_penetration
        top2_width = box_size[0] - grasp_penetration
        bottom1_width = box_size[1] - grasp_penetration
        bottom2_width = box_size[0] - grasp_penetration
        left1_width = box_size[2] - grasp_penetration
        left2_width = box_size[0] - grasp_penetration
        right1_width = box_size[2] - grasp_penetration
        right2_width = box_size[0] - grasp_penetration
        front1_width = box_size[2] - grasp_penetration
        front2_width = box_size[1] - grasp_penetration
        back1_width = box_size[2] - grasp_penetration
        back2_width = box_size[1] - grasp_penetration
        
        top1_grasp_poses = np.array([
            np.concatenate((fingertip, top1_quat, [top1_width])) for fingertip in top_fingertips1
        ])
        top2_grasp_poses = np.array([
            np.concatenate((fingertip, top2_quat, [top2_width])) for fingertip in top_fingertips2
        ])
        bottom1_grasp_poses = np.array([
            np.concatenate((fingertip, bottom1_quat, [bottom1_width])) for fingertip in bottom_fingertips1
        ])
        bottom2_grasp_poses = np.array([
            np.concatenate((fingertip, bottom2_quat, [bottom2_width])) for fingertip in bottom_fingertips2
        ])
        left1_grasp_poses = np.array([
            np.concatenate((fingertip, left1_quat, [left1_width])) for fingertip in left_fingertips1
        ])
        left2_grasp_poses = np.array([
            np.concatenate((fingertip, left2_quat, [left2_width])) for fingertip in left_fingertips2
        ])
        right1_grasp_poses = np.array([
            np.concatenate((fingertip, right1_quat, [right1_width])) for fingertip in right_fingertips1
        ])
        right2_grasp_poses = np.array([
            np.concatenate((fingertip, right2_quat, [right2_width])) for fingertip in right_fingertips2
        ])
        front1_grasp_poses = np.array([
            np.concatenate((fingertip, front1_quat, [front1_width])) for fingertip in front_fingertips1
        ])
        front2_grasp_poses = np.array([
            np.concatenate((fingertip, front2_quat, [front2_width])) for fingertip in front_fingertips2
        ])
        back1_grasp_poses = np.array([
            np.concatenate((fingertip, back1_quat, [back1_width])) for fingertip in back_fingertips1
        ])
        back2_grasp_poses = np.array([
            np.concatenate((fingertip, back2_quat, [back2_width])) for fingertip in back_fingertips2
        ])

        # Concatenate all grasp poses
        tip_grasp_poses = np.concatenate([
            top1_grasp_poses,
            top2_grasp_poses,
            bottom1_grasp_poses,
            bottom2_grasp_poses,
            left1_grasp_poses,
            left2_grasp_poses,
            right1_grasp_poses,
            right2_grasp_poses,
            front1_grasp_poses,
            front2_grasp_poses,
            back1_grasp_poses,
            back2_grasp_poses
        ])

        max_width = config["max_grasp_width"]
        # tip_grasp_poses should only contain poses with width <= max_width
        tip_grasp_poses = tip_grasp_poses[tip_grasp_poses[:, 7] <= max_width]

        SE3_ITip = su.pose7_to_SE3(tip_grasp_poses[:, :7])  # Convert tip poses to SE3
        SE3_TipHand = su.pose7_to_SE3(pose7_TipHand)  # Convert tip hand pose to SE3
        SE3_IHand = SE3_ITip @ SE3_TipHand  # Convert to hand poses
        # Convert SE3 to pose7 format (x, y, z, qw, qx, qy, qz, width)
        self.grasp_poses_in_item_frame = su.SE3_to_pose7(SE3_IHand)  # (N, 8)
        self.grasp_poses_in_item_frame = np.concatenate(
            (self.grasp_poses_in_item_frame, tip_grasp_poses[:, 7:]), axis=1
        )  # Concatenate width from tip grasp poses

        self.SE3_TipHand = SE3_TipHand  # Store the tip hand pose for later use
        # print out summary
        print(f"BoxGraspSampler initialized with {self.tip_probing_poses.shape[0]} probing poses and {self.grasp_poses_in_item_frame.shape[0]} grasp poses.")

    def sample_hand_poses(self, SO3_nominal_hand):
        """
        Sample hand poses for the box.
        
        Args:
            SO3_nominal_hand: (3, 3), hand rotation matrix in the obj local frame
        """
        assert SO3_nominal_hand.shape == (3, 3), f"SO3_nominal_hand should be of shape (3, 3), but got {SO3_nominal_hand.shape}"

        # Filter probing poses
        vecs_Y = - self.tip_probing_poses[:, 3:6]
        vec_Y_nominal_gripper = SO3_nominal_hand[:, 1] # -Y is assumed to be the gripper axis (finger to palm)
        vec_Y_nominal_gripper = vec_Y_nominal_gripper / np.linalg.norm(vec_Y_nominal_gripper)
        dot = vecs_Y @ vec_Y_nominal_gripper
        angles = np.arccos(np.clip(dot, -1.0, 1.0))
        valid_probing_poses = self.tip_probing_poses[angles < self.robot_tilting_tol_rad]
        probing_hand_poses = np.array([])
        if valid_probing_poses.shape[0] > 0:
            probing_Y = - valid_probing_poses[:, 3:6]
            probing_X = np.cross(probing_Y, SO3_nominal_hand[:, 2]) # Y x Z = X
            probing_X = probing_X / np.linalg.norm(probing_X, axis=1, keepdims=True)
            probing_Z = np.cross(probing_X, probing_Y)
            probing_Z = probing_Z / np.linalg.norm(probing_Z, axis=1, keepdims=True)
            probing_SO3 = su.transpose(np.stack((probing_X, probing_Y, probing_Z), axis=1))
            probing_quats = su.SO3_to_quat(probing_SO3) # (N, 4)

            # probing_tip_poses = np.concatenate((valid_probing_poses[:, :3], probing_quats, valid_probing_poses[:, 6:]), axis=1)
            pose7_ITip_probing = np.concatenate((valid_probing_poses[:, :3], probing_quats), axis=1)
            SE3_ITip_probing = su.pose7_to_SE3(pose7_ITip_probing)  # Convert probing poses to SE3
            SE3_IHand_probing = SE3_ITip_probing @ self.SE3_TipHand  # Convert to hand poses
            probing_hand_poses = su.SE3_to_pose7(SE3_IHand_probing)
            probing_hand_poses = np.concatenate(
                (probing_hand_poses, valid_probing_poses[:, 6:]), axis=1
            )

        # Filter grasp poses
        SO3_grasp_poses = su.quat_to_SO3(self.grasp_poses_in_item_frame[:, 3:7])
        vec_Y_grasp_poses = SO3_grasp_poses[:, 1]
        dot_grasp = vec_Y_grasp_poses @ vec_Y_nominal_gripper
        angles_grasp = np.arccos(np.clip(dot_grasp, -1.0, 1.0))
        valid_grasp_poses = self.grasp_poses_in_item_frame[angles_grasp < self.robot_tilting_tol_rad]

        # symmetry: choose the grasp with the smallest angle to the nominal
        vec_Z_nominal_gripper = SO3_nominal_hand[:, 2] # Z is assumed to be perpendicular to the grasp plane
        vec_Z_nominal_gripper = vec_Z_nominal_gripper / np.linalg.norm(vec_Z_nominal_gripper)
        R180y = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])
        for i in range(valid_grasp_poses.shape[0]):
            quat = valid_grasp_poses[i, 3:7]
            SO3_i = su.quat_to_SO3(quat)
            vec_Z_i = SO3_i[:, 2]
            dot_z = vec_Z_i @ vec_Z_nominal_gripper
            angle_z = np.arccos(np.clip(dot_z, -1.0, 1.0))
            if angle_z > np.pi / 2.0:
                # flip the grasp
                SO3_i_flipped = SO3_i @ R180y
                quat_flipped = su.SO3_to_quat(SO3_i_flipped)
                valid_grasp_poses[i, 3:7] = quat_flipped

        # # concatenate with grasp poses
        # result = np.concatenate((probing_hand_poses, valid_grasp_poses), axis=0)
        return valid_grasp_poses, probing_hand_poses


if __name__ == "__main__":
    # Example usage
    box_size = np.array([0.5, 0.5, 0.5])  # Example box size
    grasp_sampler_config = {
        "np_per_edge": [2, 2, 2],
        "probing_altitude_degrees": [45.0],  # Altitude angles in degrees
        "probing_longitude_num": 4,  # Number of longitude angles
        "edge_buffer": 0.05,
        "grasp_edge_buffer": 0.05,
        "robot_tilting_tol_deg": 90.0,  # filter based on nominal robot pose. > 180 to disable this filter
    }
    box_size = [0.3, 0.3, 0.3]
    pose7_TipHand = np.array([0, -0.222, 0, 1, 0, 0, 0])  # Example pose of the gripper tip in the hand frame

    sampler = BoxGraspSampler(box_size, pose7_TipHand, grasp_sampler_config)
    SO3_nominal_hand = np.eye(3)  # Example nominal hand orientation
    sampled_poses = sampler.sample_hand_poses(SO3_nominal_hand)
    print("Sampled hand poses shape:", sampled_poses.shape)
    print("Sampled hand poses:", sampled_poses[:5])  # Print first