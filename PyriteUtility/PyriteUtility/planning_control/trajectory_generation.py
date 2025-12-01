import toppra as ta
import toppra.constraint as constraint
import toppra.algorithm as algo
import numpy as np
from scipy.spatial.transform import Rotation as R

from spatialmath import SE3
from spatialmath import SO3

# https://docs.python.org/3/howto/logging.html
ta.setup_logging("WARNING")


class Motion:
    """
    A Motion object is a continuous trajectory that can be sampled from.
    """

    def __call__(self, time_point):
        raise NotImplementedError()

    @property
    def duration(self):
        raise NotImplementedError()


class SE3Traj(Motion):
    """
    Represent a trajectory in SE3 space.
    Returns (..., 4, 4) numpy array.
    This class uses RPY angles to represent the orientation, which can be buggy for large rotations.
    """

    def __init__(self, joint_trajectory):
        self._joint_trajectory = joint_trajectory

    def __call__(self, time_point):
        assert time_point >= 0
        js_wp = self._joint_trajectory(time_point)
        SE3_pose = SE3.Rt(
            SO3.RPY(js_wp[3], js_wp[4], js_wp[5]), [js_wp[0], js_wp[1], js_wp[2]]
        ).data[0]
        return np.array(SE3_pose)

    def at(self, times):
        if isinstance(times, np.ndarray):
            assert len(times.shape) == 1
        js_waypoints = self._joint_trajectory(times)
        N = len(times)
        SE3_trajectory = [
            SE3.Rt(SO3.RPY(js[3], js[4], js[5]), [js[0], js[1], js[2]]).data[0]
            for js in js_waypoints
        ]
        return np.array(SE3_trajectory)

    @property
    def duration(self):
        return self._joint_trajectory.duration


class JSStaticTraj(Motion):
    def __init__(self, jpos, duration):
        self._jpos = jpos
        self._duration = duration

    def __call__(self, time_point):
        return self._jpos

    @property
    def duration(self):
        return self._duration

    @duration.setter
    def duration(self, duration):
        self._duration = duration


class TSStaticTraj(Motion):
    def __init__(self, SE3, duration):
        self._SE3 = SE3
        self._duration = duration

    def __call__(self, time_point):
        return self._SE3

    @property
    def duration(self):
        return self._duration

    @duration.setter
    def duration(self, duration):
        self._duration = duration


class MotionPlan:
    """
    Represent a list of motions.
    Provide interface to sample from them given a time.
    """

    def __init__(self, motions):
        self._motions = motions
        self._total_duration = 0
        self._durations = []
        self._N = len(motions)
        for id in range(self._N):
            self._durations.append(motions[id].duration)
            self._total_duration += motions[id].duration

    @property
    def duration(self):
        return self._total_duration

    def __call__(self, time_point):
        duration_previous = 0
        for id in range(self._N):
            if duration_previous + self._durations[id] >= time_point:
                return self._motions[id](time_point - duration_previous)
            duration_previous += self._durations[id]
        raise ValueError(
            f"Query timepoint {time_point} is greater than the total duration {self._total_duration}."
        )


##
## Re-time the input trajectory to minimize duration while satisfying the constraints
##
## :param      time_stamps:            (N,) numpy array, the initial time stamps
## :param      SE3_waypoints:          a list of N SE3 waypoints
## :param      p_velocity_limits:      (3,) The translational velocity limits
## :param      p_acceleration_limits:  (3,) The translational acceleration limits
## :param      R_velocity_limits:      (3,) rad/s limits on rpy angles
## :param      R_acceleration_limits:  (3,) rad/s^2 limits on rpy angles
##
def task_space_trajectory_generation(
    time_stamps,
    SE3_waypoints,
    p_velocity_limits,
    p_acceleration_limits,
    R_velocity_limits,
    R_acceleration_limits,
):
    # data shape verification
    N = len(time_stamps)
    assert time_stamps.shape == (N,)
    assert len(SE3_waypoints) == N
    assert p_velocity_limits.shape == (3,)
    assert p_acceleration_limits.shape == (3,)
    assert R_velocity_limits.shape == (3,)
    assert R_acceleration_limits.shape == (3,)

    js_waypoints = [
        [pose.x, pose.y, pose.z, pose.rpy()[0], pose.rpy()[1], pose.rpy()[2]]
        for pose in SE3_waypoints
    ]

    for pose in SE3_waypoints:
        print("[Trajectory generation] rpy: ", pose.rpy())

    js_waypoints = np.array(js_waypoints)

    velocity_limits = np.hstack((p_velocity_limits, R_velocity_limits))
    acceleration_limits = np.hstack((p_acceleration_limits, R_acceleration_limits))

    js_traj = joint_space_trajectory_generation(
        time_stamps, js_waypoints, velocity_limits, acceleration_limits
    )

    return SE3Traj(js_traj)


def joint_space_trajectory_generation(
    time_stamps, js_waypoints, velocity_limits, acceleration_limits
):
    # data shape verification
    assert len(time_stamps.shape) == 1
    assert len(js_waypoints.shape) == 2
    assert len(velocity_limits.shape) == 1
    assert len(acceleration_limits.shape) == 1

    N = len(time_stamps)
    assert js_waypoints.shape[0] == N
    D = js_waypoints.shape[1]
    assert velocity_limits.shape[0] == D
    assert acceleration_limits.shape[0] == D

    path = ta.SplineInterpolator(time_stamps, js_waypoints)
    pc_vel = constraint.JointVelocityConstraint(velocity_limits)
    pc_acc = constraint.JointAccelerationConstraint(acceleration_limits)

    instance = algo.TOPPRA([pc_vel, pc_acc], path, parametrizer="ParametrizeConstAccel")
    jnt_traj = instance.compute_trajectory()
    return jnt_traj
