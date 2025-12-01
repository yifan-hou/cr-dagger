import numpy as np
import scipy.spatial.transform as scipy_transform

"""
This is a library of spatial math utilities.
All interfaces are numpy array based, and the last dimension(s) is always the vector dimension.

p: 3D vector representing position
SE3: 4x4 matrix representing rigid body transformation
SO3, or R: 3x3 matrix representing rotation
quat: 4D vector representing quaternion, (w, x, y, z)
twist: 6D vector representing a twist
pose7: 7D vector representing a pose, (x, y, z, qw, qx, qy, qz)
pose9: 9D vector representing a pose, (x, y, z, r1, r2, r3, r4, r5, r6)
rot6: 6D vector representing a rotation, (r1, r2, r3, r4, r5, r6)
"""

# Basic operations
def dist_quats(quat_a, quat_b):
    """
    Compute the distance between two quaternions.
    The distance is defined as the angle between the two quaternions.

    :param      quat_a:  (..., 4) np array, one or more quaternion vectors
    :param      quat_b:  (..., 4) or (4,) np array, another quaternion vector
    :return:     (...,), distance between the two quaternions
    """
    assert quat_a.shape[-1] == 4, f"quat_a should be (..., 4), got {quat_a.shape}"
    assert quat_b.shape[-1] == 4, f"quat_b should be (..., 4) or (4,), got {quat_b.shape}"

    n = quat_a.shape[:-1]  # batch dimensions
    if quat_b.ndim == 1:
        quat_b = np.broadcast_to(quat_b, n + (4,))
    
    # Normalize quaternions
    quat_a = normalize(quat_a)
    quat_b = normalize(quat_b)

    # Compute the angle between the two quaternions
    dot_product = np.sum(quat_a * quat_b, axis=-1)
    cos_value = np.clip(2*dot_product*dot_product - 1, -1.0, 1.0)  # Ensure the value is within the valid range for arccos
    return np.arccos(cos_value)
    

def dist_pose7(pose7_a, pose7_b, rot_weight):
    """
    Compute the distance between two SE3 poses.
    The distance is defined as the Euclidean distance in 3D space.

    :param      pose7_a:  (..., 7) np array, one or more pose7 vectors
    :param      pose7_b:  (7,) np array, another pose7 vector
    :param      rot_weight:    float, the weight to scale the rotational distance
    :return:     (...,), distance between the two poses
    """
    assert pose7_a.shape[-1] == 7, f"pose7_a should be (..., 7), got {pose7_a.shape}"
    assert pose7_b.shape[-1] == 7, f"pose7_b should be (7,), got {pose7_b.shape}"
    
    n = pose7_a.shape[:-1]  # batch dimensions
    if pose7_b.ndim == 1:
        pose7_b = np.broadcast_to(pose7_b, n + (7,))
    
    # Extract positions and quaternions
    pos_a  = pose7_a[..., :3]
    pos_b  = pose7_b[..., :3]
    quat_a = pose7_a[..., 3:]
    quat_b = pose7_b[..., 3:]

    # Compute translational distance
    trans_dist = np.linalg.norm(pos_a - pos_b, axis=-1)

    # Compute rotational distance
    rot_dist = dist_quats(quat_a, quat_b)

    return trans_dist + rot_dist * rot_weight

def dist_n_pose7(pose7_a, pose7_b, rot_weight):
    """
    Compute the distance between two groups of SE3 poses.
    The distance is defined as the Euclidean distance in 3D space.

    :param      pose7_a:  (..., n, 7) np array, one or more groups of n pose7 vectors
    :param      pose7_b:  (n, 7) np array, another group of n pose7 vectors
    :param      rot_weight:    float, the weight to scale the rotational distance
    :return:     (..., n), distance between the two groups of poses
    """
    assert pose7_a.shape[-1] == 7, f"pose7_a should be (..., 7), got {pose7_a.shape}"
    assert pose7_b.shape[-1] == 7, f"pose7_b should be (7,), got {pose7_b.shape}"
    assert pose7_a.ndim >= 2, f"pose7_a should be (..., n, 7), got {pose7_a.shape}"
    assert pose7_b.ndim == 2, f"pose7_b should be (n, 7), got {pose7_b.shape}"
    assert pose7_a.shape[-2] == pose7_b.shape[-2], f"pose7_a and pose7_b should have the same n, got {pose7_a.shape} and {pose7_b.shape}"

    if pose7_a.ndim > 2:
        batch_dims = pose7_a.shape[:-2]  # batch dimensions
        pose7_b = np.broadcast_to(pose7_b, batch_dims + pose7_b.shape)  # broadcast pose7_b to match batch dimensions
    
    # Extract positions and quaternions
    pos_a  = pose7_a[..., :3] # (..., n, 3)
    pos_b  = pose7_b[..., :3]
    quat_a = pose7_a[..., 3:] # (..., n, 4)
    quat_b = pose7_b[..., 3:]

    # Compute translational distance
    trans_dist = np.linalg.norm(pos_a - pos_b, axis=-1) # (..., n)

    # Compute rotational distance
    rot_dist = dist_quats(quat_a, quat_b) # (..., n)

    return trans_dist + rot_dist * rot_weight

def transpose(mat):
    """
    Transpose the last two dimensions of a batch of matrices.

    :param      mat:  (..., h, w) np array
    :return:     (...,  w, h) np array, transposed matrices
    """
    return np.swapaxes(mat, -1, -2)


def normalize(vec, eps=1e-12):
    """
    Normalize vectors along the last dimension.
    args:
        vec: (..., N) np array
    return:
        (..., N) np array of the same shape
    """
    norm = np.linalg.norm(vec, axis=-1)  # (...)
    norm = np.maximum(norm, eps)
    out = vec / norm[..., np.newaxis]
    return out


# type specific operations


def wedge3(vec):
    """
    Compute the skew-symmetric wedge matrix of a batch of 3D vectors.

    :param      vec:  (..., 3) np array
    :return:     (..., 3, 3) np array, skew-symmetric matrix
    """
    shape = vec.shape[:-1]
    out = np.zeros(shape + (3, 3), dtype=vec.dtype)
    out[..., 0, 1] = -vec[..., 2]
    out[..., 0, 2] = vec[..., 1]
    out[..., 1, 2] = -vec[..., 0]

    out[..., 1, 0] = -out[..., 0, 1]
    out[..., 2, 0] = -out[..., 0, 2]
    out[..., 2, 1] = -out[..., 1, 2]
    return out


def wedge6(vec):
    """
    Compute the homogeneous coordinates of a batch of twists.

    :param      vec:  (..., 6) np array, (v, w)
    :return:     (..., 4, 4) np array
    """
    shape = vec.shape[:-1]
    out = np.zeros(shape + (4, 4), dtype=vec.dtype)
    out[..., :3, :3] = wedge3(vec[..., 3:])
    out[..., :3, 3] = vec[..., :3]
    return out

def rotation_magnitude(R):
    # Ensure the matrix is a numpy array
    R = np.array(R)
    
    # Calculate the trace
    trace = np.trace(R)
    
    # Compute the rotation angle
    angle_rad = np.arccos((trace - 1) / 2)

    return angle_rad


def SE3_inv(mat):
    """
    Efficient inverse of a batch of SE3 matrices.

    Tested by generating random SE3 and verify SE3_inv(SE3) @ SE3 = Identity.

    :param      mat:  (..., 4, 4) np array
    :return:     (..., 4, 4) np array, inverse of the input matrix
    """
    SE3_inv = np.zeros_like(mat)
    SE3_inv[..., :3, :3] = transpose(mat[..., :3, :3])

    temp = -SE3_inv[..., :3, :3] @ np.expand_dims(mat[..., :3, 3], -1)
    SE3_inv[..., :3, 3] = temp.squeeze()
    SE3_inv[..., 3, 3] = 1
    return SE3_inv

def SO3_inv(mat):
    """
    Efficient inverse of a batch of SO3 matrices.

    :param      mat:  (..., 3, 3) np array
    :return:     (..., 3, 3) np array, inverse of the input matrix
    """
    return transpose(mat)  # SO3 is orthogonal, so the inverse is the transpose

def SE3_interp(SE3_a, SE3_b, alpha):
    """
    Interpolate between two SE3 matrices.

    :param      SE3_a:  (4, 4) np array, start SE3 matrices
    :param      SE3_b:  (4, 4) np array, end SE3 matrices
    :param      alpha:  (n,) np array, interpolation factor in [0, 1]
    :return:     (n, 4, 4) np array, interpolated SE3 matrices
    """
    assert SE3_a.shape == (4, 4), f"SE3_a should be (4, 4), got {SE3_a.shape}"
    assert SE3_b.shape == (4, 4), f"SE3_b should be (4, 4), got {SE3_b.shape}"
    assert len(alpha.shape) == 1, f"alpha should be (n,), got {alpha.shape}"

    # Extract rotation and translation components
    R_a = SE3_a[:3, :3]
    R_b = SE3_b[:3, :3]
    t_a = SE3_a[:3, 3:]
    t_b = SE3_b[:3, 3:]

    # Interpolate rotation using Slerp
    R_a_scipy = scipy_transform.Rotation.from_matrix(R_a)
    R_b_scipy = scipy_transform.Rotation.from_matrix(R_b)
    slerp = scipy_transform.Slerp([0, 1], scipy_transform.Rotation.concatenate([R_a_scipy, R_b_scipy]))
    R_interp = slerp(alpha).as_matrix().reshape(-1, 3, 3)

    # Interpolate translation linearly
    t_interp = (1 - alpha) * t_a + alpha * t_b


    # Combine rotation and translation into SE3
    SE3_interp = np.zeros((len(alpha), 4, 4), dtype=SE3_a.dtype)
    SE3_interp[:, :3, :3] = R_interp
    SE3_interp[:, :3, 3:] = t_interp.T
    
    return SE3_interp

def pose7_interp(pose7_a, pose7_b, alpha):
    """
    Interpolate between two pose7 vectors.

    :param      pose7_a:  (7,) np array, start pose7 vector
    :param      pose7_b:  (7,) np array, end pose7 vector
    :param      alpha:  (n,) np array, interpolation factor in [0, 1]
    :return:     (n, 7) np array, interpolated pose7 vectors
    """
    assert pose7_a.shape == (7,), f"pose7_a should be (7,), got {pose7_a.shape}"
    assert pose7_b.shape == (7,), f"pose7_b should be (7,), got {pose7_b.shape}"
    assert len(alpha.shape) <= 1, f"alpha should be (n,), got {alpha.shape}"

    # Extract position and quaternion components
    pos_a = pose7_a[:3]
    pos_b = pose7_b[:3]
    quat_a = pose7_a[3:]
    quat_b = pose7_b[3:]

    # Interpolate position linearly
    pos_interp = (1 - alpha[:, np.newaxis]) * pos_a + alpha[:, np.newaxis] * pos_b

    # Interpolate quaternion using Slerp
    quat_a_scipy = scipy_transform.Rotation.from_quat(quat_a, scalar_first=True)
    quat_b_scipy = scipy_transform.Rotation.from_quat(quat_b, scalar_first=True)
    slerp = scipy_transform.Slerp([0, 1], scipy_transform.Rotation.concatenate([quat_a_scipy, quat_b_scipy]))
    quat_interp = slerp(alpha).as_quat(scalar_first=True)

    # Combine position and quaternion into pose7
    pose7_interp = np.zeros((len(alpha), 7), dtype=pose7_a.dtype)
    pose7_interp[:, :3] = pos_interp
    pose7_interp[:, 3:] = quat_interp

    return pose7_interp


# transformations
def trans_p_by_SE3(p, SE3):
    """
    Transform a batch of 3D points by a batch of SE3 matrices.

    :param      p:    (..., 3) np array, 3D points
    :param      SE3:  (..., 4, 4) np array, SE3 matrices
    :return:     (..., 3) np array, transformed points
    """
    p = np.expand_dims(p, -2)
    return np.sum(SE3[..., :3, :3] * p, axis=-1) + SE3[..., :3, 3]


# Type conversions


def JacTwist2BodyV(R):
    """
    From a SO3, compute the Jacobian matrix that maps twist to body velocity.

    :param      R:  (3, 3) np array, the rotation matrix
    :return:     (6, 6) np array, the Jacobian matrix
    """

    Jac = np.eye(6)
    Jac[3, 3] = R[0, 2] * R[0, 2] + R[1, 2] * R[1, 2] + R[2, 2] * R[2, 2]
    Jac[3, 5] = -R[0, 0] * R[0, 2] - R[1, 0] * R[1, 2] - R[2, 0] * R[2, 2]
    Jac[4, 3] = -R[0, 0] * R[0, 1] - R[1, 0] * R[1, 1] - R[2, 0] * R[2, 1]
    Jac[4, 4] = R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0] + R[2, 0] * R[2, 0]
    Jac[5, 4] = -R[0, 1] * R[0, 2] - R[1, 1] * R[1, 2] - R[2, 1] * R[2, 2]
    Jac[5, 5] = R[0, 1] * R[0, 1] + R[1, 1] * R[1, 1] + R[2, 1] * R[2, 1]

    return Jac


def pose7_to_SE3(pose7):
    """
    pose7: [:, 7] with x,y,z,qw,qx,qy,qz
    returns: [:, 4, 4] SE3 matrices
    """
    # normalize quaternion
    quat = pose7[..., 3:] / np.linalg.norm(pose7[..., 3:], axis=-1, keepdims=True)

    qw = quat[..., 0]
    qx = quat[..., 1]
    qy = quat[..., 2]
    qz = quat[..., 3]
    q11 = qx * qx
    q22 = qy * qy
    q33 = qz * qz
    q01 = qw * qx
    q02 = qw * qy
    q03 = qw * qz
    q12 = qx * qy
    q13 = qx * qz
    q23 = qy * qz

    shape = pose7.shape[:-1]
    SE3 = np.zeros(shape + (4, 4), dtype=pose7.dtype)
    SE3[..., 0, 0] = 1.0 - 2.0 * q22 - 2.0 * q33
    SE3[..., 0, 1] = 2.0 * (q12 - q03)
    SE3[..., 0, 2] = 2.0 * (q13 + q02)
    SE3[..., 1, 0] = 2.0 * (q12 + q03)
    SE3[..., 1, 1] = 1.0 - 2.0 * q11 - 2.0 * q33
    SE3[..., 1, 2] = 2.0 * (q23 - q01)
    SE3[..., 2, 0] = 2.0 * (q13 - q02)
    SE3[..., 2, 1] = 2.0 * (q23 + q01)
    SE3[..., 2, 2] = 1.0 - 2.0 * q11 - 2.0 * q22

    SE3[..., :3, 3] = pose7[..., :3]

    SE3[..., 3, 3] = 1

    return SE3


def pose9_to_SE3(d9):
    p = d9[..., :3]
    d6 = d9[..., 3:]
    R = rot6_to_SO3(d6)
    SE3 = np.zeros(d9.shape[:-1] + (4, 4), dtype=d9.dtype)
    SE3[..., :3, :3] = R
    SE3[..., :3, 3] = p
    SE3[..., 3, 3] = 1
    return SE3

def aa_to_quat(axis, angle):
    """
    Convert axis-angle representation to a quaternion.

    :param      axis:  (..., 3) np array of unit vectors, rotation axis
    :param      angle: (..., 1) np array, rotation angle
    :return:     (4,) np array, quaternion (w, x, y, z)
    """
    assert axis.shape[-1] == 3, f"axis should be (..., 3), got {axis.shape}"
    assert angle.shape[-1] == 1, f"angle should be (..., 1), got {angle.shape}"

    axis = normalize(axis)
    half_angle = angle / 2
    sin_half_angle = np.sin(half_angle)
    q = np.zeros(axis.shape[:-1] + (4,), dtype=axis.dtype)
    q[..., 0] = np.cos(np.squeeze(half_angle)) # squeeze change (..., 1) to (...,)
    q[..., 1:] = axis * sin_half_angle
    return q

def quat_to_aa(quat, tol=1e-7):
    """
    (not vectorized)
    Convert a quaternion to axis-angle representation.

    :param      quat:  (4,) np array
    :return:     (4,) np array, axis-angle representation (axis, angle)
    """
    angle = 2 * np.arccos(quat[..., 0])

    axis = quat[1:]
    axis_norm = np.linalg.norm(axis)

    if axis_norm < tol:
        return np.array([1, 0, 0, 0], dtype=quat.dtype)
    axis /= axis_norm

    return np.array([*axis, angle], dtype=quat.dtype)

def quat_to_SO3(quat):
    """
    Convert a quaternion to a rotation matrix (SO3).

    :param      quat:  (..., 4) np array, quaternion in (w, x, y, z) format
    :return:     (..., 3, 3) np array, rotation matrix
    """
    assert quat.shape[-1] == 4, f"quat should be (..., 4), got {quat.shape}"

    qw = quat[..., 0]
    qx = quat[..., 1]
    qy = quat[..., 2]
    qz = quat[..., 3]
    q11 = qx * qx
    q22 = qy * qy
    q33 = qz * qz
    q01 = qw * qx
    q02 = qw * qy
    q03 = qw * qz
    q12 = qx * qy
    q13 = qx * qz
    q23 = qy * qz

    shape = quat.shape[:-1]
    SO3 = np.zeros(shape + (3, 3), dtype=quat.dtype)
    SO3[..., 0, 0] = 1.0 - 2.0 * q22 - 2.0 * q33
    SO3[..., 0, 1] = 2.0 * (q12 - q03)
    SO3[..., 0, 2] = 2.0 * (q13 + q02)
    SO3[..., 1, 0] = 2.0 * (q12 + q03)
    SO3[..., 1, 1] = 1.0 - 2.0 * q11 - 2.0 * q33
    SO3[..., 1, 2] = 2.0 * (q23 - q01)
    SO3[..., 2, 0] = 2.0 * (q13 - q02)
    SO3[..., 2, 1] = 2.0 * (q23 + q01)
    SO3[..., 2, 2] = 1.0 - 2.0 * q11 - 2.0 * q22

    return SO3

def rot6_to_SO3(d6):
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = normalize(a1)
    b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = normalize(b2)
    b3 = np.cross(b1, b2, axis=-1)
    SO3 = np.stack((b1, b2, b3), axis=-2)
    return SO3


def SE3_to_adj(SE3):
    """
    Compute the adjoint matrix of a batch of SE3 matrices.

    :param      SE3:  (..., 4, 4) np array
    :return:     (..., 6, 6) np array, adjoint matrices
    """
    shape = SE3.shape[:-2]
    R = SE3[..., :3, :3]
    p = SE3[..., :3, 3]
    Adj = np.zeros(shape + (6, 6), dtype=SE3.dtype)
    Adj[..., :3, :3] = R
    Adj[..., :3, 3:] = wedge3(p) @ R
    Adj[..., 3:, 3:] = R
    return Adj


def SE3_to_pose7(SE3):
    p = SE3[..., :3, 3]
    R = SE3[..., :3, :3]
    q = SO3_to_quat(R)
    pose7 = np.concatenate([p, q], axis=-1)
    return pose7


def SE3_to_pose9(SE3):
    p = SE3[..., :3, 3]
    R = SE3[..., :3, :3]
    d6 = SO3_to_rot6d(R)
    pose9 = np.concatenate([p, d6], axis=-1)
    return pose9


def SE3_to_se3(SE3, kEpsilon=1e-7):
    """displacement to twist coordinate (se3)
    :param      SE3:  (4, 4) np array
    :return      (6) np array, twist coordinates
    """
    assert SE3.shape == (4, 4)
    p = SE3[:3, 3]
    R = SE3[:3, :3]
    omega = SO3_to_so3(R, kEpsilon)
    theta = np.linalg.norm(omega, axis=-1, keepdims=True)
    if theta < kEpsilon:
        return np.concatenate([p, omega], axis=-1)
    omega /= theta
    M = (np.eye(3) - R) @ wedge3(omega) + omega @ omega.T * theta
    se3 = np.zeros(SE3.shape[:-2] + (6,), dtype=SE3.dtype)
    se3[:3] = np.linalg.solve(M, p)
    se3[3:] = omega
    se3 *= theta
    return se3


def SE3_to_spt(SE3, kEpsilon=1e-7):
    """displacement to special twist coordinate
    :param      SE3:  (..., 4, 4) np array
    :return      (..., 6) np array, twist coordinates
    """
    twist_coordinate = np.zeros(SE3.shape[:-2] + (6,), dtype=SE3.dtype)
    twist_coordinate[..., :3] = SE3[..., :3, 3]
    twist_coordinate[..., 3:] = SO3_to_so3(SE3[..., :3, :3], kEpsilon)
    return twist_coordinate


def se3_to_SE3(se3, kEpsilon=1e-9):
    """twist coordinate to displacement
    :param      se3:  (6,) np array, twist coordinates
    :return      (4, 4) np array
    """
    if se3.shape == (6, 1):
        se3 = se3.reshape(6)  # (6,1) -> (6,)
    if se3.shape != (6,):
        raise ValueError(f"se3 shape should be (6, 1) or (6,), got {se3.shape}")

    v = se3[:3]
    w = se3[3:]
    theta = np.linalg.norm(w)

    if np.fabs(theta) < kEpsilon:
        SE3 = np.eye(4)
        SE3[:3, 3] = v
    else:
        v /= theta
        w /= theta
        R = so3_to_SO3(w)
        SE3 = np.eye(4)
        SE3[:3, :3] = R
        SE3[:3, 3] = (np.eye(3) - R) @ np.cross(w, v) + w * w.T @ v * theta
    return SE3


def SO3_to_rot6d(mat):
    batch_dim = mat.shape[:-2]
    out = mat[..., :2, :].copy().reshape(batch_dim + (6,))
    return out


def SO3_to_so3(R, kEpsilon=1e-7):
    """Get exponential coordinate of a rotation matrix

    :param      R:    (3, 3) numpy array

    :returns:   (3,) numpy array
    """
    assert R.shape == (3, 3)
    dim = len(R.shape)
    output_shape = R.shape[:-2] + (3,)
    temp_arg_to_cos = (np.trace(R, axis1=dim - 2, axis2=dim - 1) - 1.0) / 2.0
    # truncate temp_arg_to_cos between  -1.0, 1.0
    temp_arg_to_cos = np.maximum(np.minimum(temp_arg_to_cos, 1), -1)
    theta = np.arccos(temp_arg_to_cos)
    if np.fabs(theta) < kEpsilon:
        so3 = np.broadcast_to([1.0, 0.0, 0.0], output_shape).copy()
    else:
        so3 = np.array(
            [
                R[..., 2, 1] - R[..., 1, 2],
                R[..., 0, 2] - R[..., 2, 0],
                R[..., 1, 0] - R[..., 0, 1],
            ]
        )
        so3 /= 2.0 * np.sin(theta)
    so3 *= theta
    return so3


def so3_to_SO3(v, kEpsilon=1e-9):
    """Get rotation matrix from exponential coordinate

    :param      v:    (..., 3) numpy array

    :returns:   (..., 3, 3) numpy array

    Needs to be tested
    """
    theta = np.linalg.norm(v, axis=-1, keepdims=True)
    theta = np.maximum(theta, kEpsilon)
    vn = v / theta
    v_wedge = wedge3(vn)
    SO3 = (
        np.eye(3)
        + v_wedge * np.sin(theta[..., np.newaxis])
        + v_wedge @ v_wedge * (1.0 - np.cos(theta[..., np.newaxis]))
    )
    return SO3


def SO3_to_quat(R):
    """Convert rotation matrix to quaternion.
    Uses scipy.spatial.transform.Rotation internally, which can handle one or an array of rotation matrices.

    :param      R:  (..., 3, 3) np array
    :return:     (..., 4) np array, quaternion
    """
    assert R.shape[-2:] == (3, 3), f"R should be (..., 3, 3), got {R.shape}"
    if len(R.shape) <= 3:
        if len(R.shape) == 3:
            assert R.shape[0] > 0, "R should have at least one matrix"
        return scipy_transform.Rotation.from_matrix(R).as_quat(scalar_first=True)
    else:
        shape = R.shape[:-2] + (4,)
        R = R.reshape(-1, 3, 3)
        q = scipy_transform.Rotation.from_matrix(R).as_quat(scalar_first=True)
        return q.reshape(shape)


def spt_to_SE3(twist, kEpsilon=1e-9):
    """special twist coordinate to displacement
    :param      twist:  (..., 6) np array, twist coordinates
    :return      (..., 4, 4) np array
    """
    if twist.shape == (6, 1):
        twist = twist.reshape(6)  # (6,1) -> (6,)
    if twist.shape[-1] != 6:
        raise ValueError(f"twist shape should be (..., 6) or (6,), got {twist.shape}")
    SE3 = np.zeros(twist.shape[:-1] + (4, 4), dtype=twist.dtype)
    SE3[..., :3, :3] = so3_to_SO3(twist[..., 3:], kEpsilon)
    SE3[..., :3, 3] = twist[..., :3]
    SE3[..., 3, 3] = 1
    return SE3


def twc_to_SE3(twc):
    """twist coordinate to displacement
    :param      twc:  (..., 6) np array, twist coordinates
    :return      (..., 4, 4) np array

    Needs to be tested
    """
    if twc.shape == (6, 1):
        twc = twc.reshape(6)  # (6,1) -> (6,)
    if twc.shape[-1] != 6:
        raise ValueError(f"twc shape should be (..., 6) or (6,), got {twc.shape}")
    SE3 = np.zeros(twc.shape[:-1] + (4, 4), dtype=twc.dtype)
    SE3[..., :3, :3] = so3_to_SO3(twc[..., 3:])
    SE3[..., :3, 3] = twc[..., :3]
    SE3[..., 3, 3] = 1
    return SE3


## Legacy code from UMI

# def pos_rot_to_mat(pos, rot):
#     shape = pos.shape[:-1]
#     mat = np.zeros(shape + (4,4), dtype=pos.dtype)
#     mat[...,:3,3] = pos
#     mat[...,:3,:3] = rot.as_matrix()
#     mat[...,3,3] = 1
#     return mat

def mat_to_pos_rot(mat):
    pos = (mat[...,:3,3].T / mat[...,3,3].T).T
    rot = scipy_transform.Rotation.from_matrix(mat[...,:3,:3]).as_rotvec()
    return pos, rot

# def pos_rot_to_pose(pos, rot):
#     shape = pos.shape[:-1]
#     pose = np.zeros(shape+(6,), dtype=pos.dtype)
#     pose[...,:3] = pos
#     pose[...,3:] = rot.as_rotvec()
#     return pose

# def pose_to_pos_rot(pose):
#     pos = pose[...,:3]
#     rot = st.Rotation.from_rotvec(pose[...,3:])
#     return pos, rot

# def pose_to_mat(pose):
#     return pos_rot_to_mat(*pose_to_pos_rot(pose))

# tests
def test():
    # Test quaternion distance
    quat_a = np.array([1, 0, 0, 0])
    quat_b = np.array([0, 1, 0, 0])
    print("Quaternion Distance:", dist_quats(quat_a, quat_b))

    # Test pose7 distance
    pose7_a = np.array([0, 0, 0, 1, 0, 0, 0])
    pose7_b = np.array([1, 1, 1, 1, 0, 0, 0])
    print("Pose7 Distance:", dist_pose7(pose7_a, pose7_b, rot_weight=1.0))

    pose7_a = np.array([[1, 0, 0, 1, 0, 0, 0],
                        [0, 2, 0, 0, 1, 0, 0],
                        [0, 0, 3, 0, 0, 1, 0]])
    print("Pose7 Distance:", dist_pose7(pose7_a, pose7_b, rot_weight=1.0))

    pose7_b = np.array([[1, 0, 0, 1, 0, 0, 0],
                        [0, 2, 0, 0, 1, 0, 0],
                        [0, 0, 3, 0, 0, 1, 0]])
    print("Pose7 Distance:", dist_pose7(pose7_a, pose7_b, rot_weight=1.0))


if __name__ == "__main__":
    test()