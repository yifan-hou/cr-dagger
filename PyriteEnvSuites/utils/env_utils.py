import numpy as np
import copy
import spatialmath as sm
import spatialmath.base as smb
from PyriteUtility.planning_control.trajectory import (
    LinearInterpolator,
    LinearTransformationInterpolator,
    CombinedGeometricPath,
)
from typing import Dict, Callable, Tuple, List
from PyriteUtility.spatial_math import spatial_utilities as su

def get_real_obs_resolution(shape_meta: dict) -> Tuple[int, int]:
    out_res = None
    obs_shape_meta = shape_meta["obs"]
    for key, attr in obs_shape_meta.items():
        type = attr.get("type", "low_dim")
        shape = attr.get("shape")
        if type == "rgb":
            co, ho, wo = shape
            if out_res is None:
                out_res = (wo, ho)
            assert out_res == (wo, ho)
    return out_res
        
def ts_to_js_traj(action_mats, time_steps, robot):
    """
    Convert an array of task space waypoints to a joint space trajectory.
    Args:
        action_mats: (T, 4, 4) array of task space waypoints.
        time_steps: (T,) array of time points for each waypoint.
        robot: MujocoRobot object with inverse_kinematics_SE3() method.
    Returns:
        A spline trajectory in joint space
    A ValueError is raised if no IK solution is found.
    """
    assert action_mats.shape[1] == 4
    assert action_mats.shape[2] == 4
    if action_mats.shape[0] != time_steps.shape[0]:
        print("action_mats.shape[0]:", action_mats.shape[0])
        print("time_steps.shape[0]:", time_steps.shape[0])
        raise ValueError(
            "The number of time steps must match the number of action matrices."
        )
    jpos_waypoints = []
    for mat in action_mats:
        q = smb.r2q(mat[:3, :3], check=False)
        q = q / np.linalg.norm(q)
        pose7 = np.concatenate([mat[:3, 3], q])
        ik_result = robot.inverse_kinematics(pose7, True)
        if ik_result is None:
            raise ValueError("No IK solution found.")
        jpos_waypoints.append(ik_result)
    return LinearInterpolator(time_steps, jpos_waypoints)

def pose9g1_to_traj(target_mats, eoat, time_steps):
    """
    Args:
        target_mats: (T, 4, 4) array of task space waypoints.
        eoat: (T, 1) array of gripper position
        time_steps: (T,) array of time points for each waypoint.
    Returns:
        A trajectory with concatenated 10-dimensional waypoints.
    """
    assert target_mats.shape[1] == 4
    assert target_mats.shape[2] == 4
    assert eoat.shape[1] == 1
    if (
        target_mats.shape[0] != time_steps.shape[0]
        or eoat.shape[0] != time_steps.shape[0]
    ):
        print("target_mats.shape[0]:", target_mats.shape[0])
        print("eoat.shape[0]:", eoat.shape[0])
        print("time_steps.shape[0]:", time_steps.shape[0])
        raise ValueError("The number of time steps must match among sources of inputs.")

    target_traj = LinearTransformationInterpolator(time_steps, target_mats)
    eoat_traj = LinearInterpolator(time_steps, eoat)

    return CombinedGeometricPath([target_traj, eoat_traj])

def js_to_traj(target_js, time_steps):
    """
    Args:
        target_js: (T, N) array of joint space waypoints.
        time_steps: (T,) array of time points for each waypoint.
    Returns:
        A trajectory with concatenated N-dimensional waypoints.
    """
    if target_js.shape[0] != time_steps.shape[0]:
        print("target_js.shape[0]:", target_js.shape[0])
        print("time_steps.shape[0]:", time_steps.shape[0])
        raise ValueError("The number of time steps must match among sources of inputs.")

    return LinearInterpolator(time_steps, target_js)

def pose9pose9s1_to_traj(target_mats, vt_mats, stiffness, time_steps):
    """
    Args:
        target_mats: (T, 4, 4) array of task space waypoints.
        vt_mats: (T, 4, 4) array of task space virtual target waypoints.
        stiffness: (T,) array of stiffness for each waypoint.
        time_steps: (T,) array of time points for each waypoint.
    Returns:
        A trajectory with concatenated 19-dimensional waypoints.
    A ValueError is raised if no IK solution is found.
    """
    assert target_mats.shape[1] == 4
    assert target_mats.shape[2] == 4
    assert vt_mats.shape[1] == 4
    assert vt_mats.shape[2] == 4
    if (
        target_mats.shape[0] != time_steps.shape[0]
        or vt_mats.shape[0] != time_steps.shape[0]
        or stiffness.shape[0] != time_steps.shape[0]
    ):
        print("target_mats.shape[0]:", target_mats.shape[0])
        print("vt_mats.shape[0]:", vt_mats.shape[0])
        print("stiffness.shape[0]:", stiffness.shape[0])
        print("time_steps.shape[0]:", time_steps.shape[0])
        raise ValueError("The number of time steps must match among sources of inputs.")

    target_traj = LinearTransformationInterpolator(time_steps, target_mats)
    vt_traj = LinearTransformationInterpolator(time_steps, vt_mats)
    stiffness_traj = LinearInterpolator(time_steps, stiffness)

    return CombinedGeometricPath([target_traj, vt_traj, stiffness_traj])

def pose9pose9s1a2_to_traj(target_mats, vt_mats, stiffness, eoat, time_steps):
    """
    Args:
        target_mats: (T, 4, 4) array of task space waypoints.
        vt_mats: (T, 4, 4) array of task space virtual target waypoints.
        stiffness: (T,) array of stiffness for each waypoint.
        eoat: (T, 2) array of gripper position and force.
        time_steps: (T,) array of time points for each waypoint.
    Returns:
        A trajectory with concatenated 21-dimensional waypoints.
    """
    assert target_mats.shape[1] == 4
    assert target_mats.shape[2] == 4
    assert vt_mats.shape[1] == 4
    assert vt_mats.shape[2] == 4
    assert eoat.shape[1] == 2
    if (
        target_mats.shape[0] != time_steps.shape[0]
        or vt_mats.shape[0] != time_steps.shape[0]
        or stiffness.shape[0] != time_steps.shape[0]
        or eoat.shape[0] != time_steps.shape[0]
    ):
        print("target_mats.shape[0]:", target_mats.shape[0])
        print("vt_mats.shape[0]:", vt_mats.shape[0])
        print("stiffness.shape[0]:", stiffness.shape[0])
        print("eoat.shape[0]:", eoat.shape[0])
        print("time_steps.shape[0]:", time_steps.shape[0])
        raise ValueError("The number of time steps must match among sources of inputs.")

    target_traj = LinearTransformationInterpolator(time_steps, target_mats)
    vt_traj = LinearTransformationInterpolator(time_steps, vt_mats)
    stiffness_traj = LinearInterpolator(time_steps, stiffness)
    eoat_traj = LinearInterpolator(time_steps, eoat)

    return CombinedGeometricPath([target_traj, vt_traj, stiffness_traj, eoat_traj])

def decode_stiffness(SE3_TW, SE3_ref_poses, SE3_vt_poses, stiffnesses,
                     default_stiffness, default_stiffness_rot,
                     target_stiffness_override):
    ts_targets_nominal = su.SE3_to_pose7(
        SE3_ref_poses.reshape([-1, 4, 4])
    )
    ts_targets_virtual = su.SE3_to_pose7(
        SE3_vt_poses.reshape([-1, 4, 4])
    )

    ts_stiffnesses = np.zeros([6, 6 * ts_targets_virtual.shape[0]])
    for i in range(ts_targets_virtual.shape[0]):
        SE3_target = SE3_ref_poses[i].reshape([4, 4])
        SE3_virtual_target = SE3_vt_poses[i].reshape([4, 4])
        stiffness = stiffnesses[i]

        # stiffness: 1. convert vt to tool frame
        SE3_TVt = SE3_TW @ SE3_virtual_target
        SE3_Ttarget = SE3_TW @ SE3_target

        # stiffness: 2. compute stiffness matrix in the tool frame
        delta_vec = (
            SE3_TVt[:3, 3] - SE3_Ttarget[:3, 3]
        ).reshape(3)

        if np.linalg.norm(delta_vec) < 0.001:  #
            compliance_direction_tool = np.array([1.0, 0.0, 0.0])
        else:
            compliance_direction_tool = delta_vec / np.linalg.norm(delta_vec)
        X = compliance_direction_tool
        Y = np.cross(X, np.array([0, 0, 1]))
        Y /= np.linalg.norm(Y)
        Z = np.cross(X, Y)

        target_stiffness = stiffness
        if target_stiffness_override is not None:
            target_stiffness = target_stiffness_override
            # recompute the virtual target given the new stiffness
            # target_displacement = np.zeros_like(delta_vec)
            # if np.linalg.norm(delta_vec) > 0.01:
            target_displacement = delta_vec * stiffness / target_stiffness_override
            # print("SE3_TW", SE3_TW)
            # print("SE3_TVt", SE3_TVt)
            # print("SE3_Ttarget", SE3_Ttarget)
            # print("target_displacement", target_displacement)
            print("stiffness", stiffness)
            # print("target_stiffness_override", target_stiffness_override)
            # exit(0)
            ts_targets_virtual[i, :3] = ts_targets_nominal[i, :3] + target_displacement

        M = np.diag(
            [target_stiffness, default_stiffness, default_stiffness]
        )
        S = np.array([X, Y, Z]).T
        stiffness_matrix = S @ M @ np.linalg.inv(S)
        stiffness_matrix_full = np.eye(6) * default_stiffness_rot
        stiffness_matrix_full[:3, :3] = stiffness_matrix

        ts_stiffnesses[:, 6 * i : 6 * i + 6] = stiffness_matrix_full
    
    return ts_targets_nominal, ts_targets_virtual, ts_stiffnesses