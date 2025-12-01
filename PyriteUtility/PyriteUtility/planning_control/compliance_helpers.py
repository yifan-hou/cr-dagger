import sys
import os

# import cvxpy as cp
import numpy as np

sys.path.append(os.path.join(sys.path[0], "../../"))

from plotly.subplots import make_subplots
import plotly.graph_objs as go
import plotly.io as pio
import plotly.express as px

pio.templates.default = "plotly_dark"
pio.renderers.default = "browser"


import PyriteUtility.spatial_math.spatial_utilities as su
import PyriteUtility.math.numerical_differentiation as nd


def estimate_stiffness(
    SE3_traj, wrench_traj, min_gap, norm_weight, translation_only=False
):
    """
    Compute stiffness from the given motion data. Assume both position and wrench
    are already converted to the same reference frame. Solves a convex optimization.

    SE3_traj: (..., 4,4) a sequence of 4x4 transformation matrices
    wrench_traj: (..., 6) a sequence of wrenches
    min_gap: the minimum gap between the pairs of indices when computing the stiffness
    norm_weight: the weight for the regularization term
    """
    if isinstance(SE3_traj, list):
        SE3_traj = np.array(SE3_traj)
        wrench_traj = np.array(wrench_traj)
    assert SE3_traj.shape[0] == wrench_traj.shape[0]
    assert SE3_traj.shape[0] > 6  # avoid degenerate case
    assert SE3_traj.shape[1] == 4
    assert SE3_traj.shape[2] == 4
    assert wrench_traj.shape[1] == 6

    N = SE3_traj.shape[0]
    # generate all combinations of pairs of indices with a minimum gap of min_gap
    pairs = [
        (i, j) for i in range(0, N, min_gap) for j in range(i + min_gap, N, min_gap)
    ]
    twist_traj = np.array(
        [su.SE3_to_spt(su.SE3_inv(SE3_traj[i]) @ SE3_traj[j]) for i, j in pairs]
    )
    delta_wrench_traj = np.array([wrench_traj[j] - wrench_traj[i] for i, j in pairs])

    # Compute stiffness, estimate a positive semi-definite matrix
    X = twist_traj
    Y = delta_wrench_traj

    # formulate a CVXPY problem
    K = cp.Variable((6, 6), PSD=True)

    if translation_only:
        K = cp.Variable((3, 3), PSD=True)
        X = X[:, :3]
        Y = Y[:, :3]

    cost = cp.norm(Y - X @ K, "fro") + norm_weight * cp.norm(K, "fro")
    prob = cp.Problem(cp.Minimize(cost))
    prob.solve()

    env_stiffness = K.value
    return env_stiffness


class StiffnessEstimator:
    """Class to handle storage of data and function call for stiffness estimation"""

    def __init__(self, window_size, min_gap=5, norm_weight=0.1):
        self.N = window_size
        self.min_gap = min_gap
        self.norm_weight = norm_weight

        self.mat_buffer = []
        self.wrench_buffer = []

    def append(self, mat, wrench):
        self.mat_buffer.append(mat)
        self.wrench_buffer.append(wrench)

        if len(self.mat_buffer) > self.N:
            self.mat_buffer.pop(0)
            self.wrench_buffer.pop(0)

    def estimate(self):
        if len(self.mat_buffer) < self.N:
            return None
        assert len(self.mat_buffer) == self.N
        assert len(self.wrench_buffer) == self.N

        return estimate_stiffness(
            self.mat_buffer,
            self.wrench_buffer,
            self.min_gap,
            self.norm_weight,
            translation_only=True,
        )


class StiffnessObserver:
    """Class to handle storage of data and function call for stiffness observation.
    This is a implementation of this paper: https://www.roboticsproceedings.org/rss06/p12.pdf
    Finite differencing formula is obtained from https://web.media.mit.edu/~crtaylor/calculator.html
    Finite difference indices:
        buffer_size = 5:    -2, -1, 0, 1, 2
    """

    def __init__(
        self,
        dt,
        alpha=0.1,
        mass=1.0,
        damping=0.1,
        k0=1.0,  # initial value of the stiffness
        yd_threshold=0.1,
        buffer_size=5,
    ):
        self.y = []
        self.f = []

        self.buffer_size = buffer_size
        self.center_id = buffer_size // 2
        self.dt = dt
        self.alpha = alpha
        self.mass = mass
        self.damping = damping
        self.yd_threshold = yd_threshold
        # estimation of stiffness
        self.k_hat = np.ones(3) * k0

        # logging, has some duplications
        self.log = {
            "y": [],
            "yd": [],
            "ydd": [],
            "yddd": [],
            "f": [],
            "fd": [],
            "f_hat": [],
            "fd_hat": [],
            "kd_hat": [],
            "k_hat": [],
        }

    def update(self, mat, wrench):
        y_new = mat[:3, 3]

        self.y.append(y_new)
        self.f.append(wrench[:3])
        if len(self.y) > self.buffer_size:
            self.y.pop(0)
            self.f.pop(0)

            yi = self.y[self.center_id]
            yi_d = nd.finite_difference(self.y, self.center_id, self.dt, 1)
            yi_dd = nd.finite_difference(self.y, self.center_id, self.dt, 2)
            yi_ddd = nd.finite_difference(self.y, self.center_id, self.dt, 3)

            fi = self.f[self.center_id]
            fi_d = nd.finite_difference(self.f, self.center_id, self.dt, 1)

            # yd_new = (y_new - self.y) / self.dt
            # ydd_new = (yd_new - self.yd) / self.dt
            # yddd_new = (ydd_new - self.ydd) / self.dt
            # fi = wrench[:3]

            yd_threshold = 1
            ydd_threshold = 10
            yddd_threshold = 100

            fd_threshold = 2000
            yi_d = np.clip(yi_d, -yd_threshold, yd_threshold)
            yi_dd = np.clip(yi_dd, -ydd_threshold, ydd_threshold)
            yi_ddd = np.clip(yi_ddd, -yddd_threshold, yddd_threshold)
            fi_d = np.clip(fi_d, -fd_threshold, fd_threshold)

            fi_hat = self.mass * yi_dd + self.damping * yi_d

            fi_d_hat = self.mass * yi_ddd + self.damping * yi_dd + self.k_hat * yi_d
            fi_d_tilde = fi_d - fi_d_hat

            yi_d_sign = 0
            if np.linalg.norm(yi_d) > self.yd_threshold:
                yi_d_sign = yi_d / np.linalg.norm(yi_d)

            k_d_hat = (
                self.alpha * fi_d_tilde * yi_d_sign
                + self.alpha * self.k_hat * np.abs(yi_d)
            )

            self.k_hat += k_d_hat * self.dt

            # logging
            self.log["y"].append(yi)
            self.log["yd"].append(yi_d)
            self.log["ydd"].append(yi_dd)
            self.log["yddd"].append(yi_ddd)
            self.log["f"].append(fi)
            self.log["fd"].append(fi_d)
            self.log["f_hat"].append(fi_hat)
            self.log["fd_hat"].append(fi_d_hat)
            self.log["kd_hat"].append(k_d_hat)
            self.log["k_hat"].append(self.k_hat.copy())

            print(
                "self.k_hat: ",
                self.k_hat,
                ", len(self.log['k_hat']): ",
                len(self.log["k_hat"]),
            )
            # return a diagonal matrix whose diagonal is the estimated stiffness
            return np.diag(self.k_hat)
        else:
            return None

    def plot_log(self, title="Figure"):
        self.log["y"] = np.array(self.log["y"])
        self.log["yd"] = np.array(self.log["yd"])
        self.log["ydd"] = np.array(self.log["ydd"])
        self.log["yddd"] = np.array(self.log["yddd"])
        self.log["f"] = np.array(self.log["f"])
        self.log["fd"] = np.array(self.log["fd"])
        self.log["f_hat"] = np.array(self.log["f_hat"])
        self.log["fd_hat"] = np.array(self.log["fd_hat"])
        self.log["kd_hat"] = np.array(self.log["kd_hat"])
        self.log["k_hat"] = np.array(self.log["k_hat"])

        x = np.arange(len(self.log["k_hat"]))
        fig = make_subplots(
            rows=8,
            cols=1,
            shared_xaxes="all",
            subplot_titles=("y", "yd", "ydd", "yddd", "f", "fd", "kd_hat", "k_hat"),
        )
        fig.add_trace(go.Scatter(x=x, y=self.log["y"][:, 0], name="y"), row=1, col=1)
        fig.add_trace(go.Scatter(x=x, y=self.log["y"][:, 1], name="y"), row=1, col=1)
        fig.add_trace(go.Scatter(x=x, y=self.log["y"][:, 2], name="y"), row=1, col=1)
        fig.add_trace(go.Scatter(x=x, y=self.log["yd"][:, 0], name="yd"), row=2, col=1)
        fig.add_trace(go.Scatter(x=x, y=self.log["yd"][:, 1], name="yd"), row=2, col=1)
        fig.add_trace(go.Scatter(x=x, y=self.log["yd"][:, 2], name="yd"), row=2, col=1)
        fig.add_trace(
            go.Scatter(x=x, y=self.log["ydd"][:, 0], name="ydd"), row=3, col=1
        )
        fig.add_trace(
            go.Scatter(x=x, y=self.log["ydd"][:, 1], name="ydd"), row=3, col=1
        )
        fig.add_trace(
            go.Scatter(x=x, y=self.log["ydd"][:, 2], name="ydd"), row=3, col=1
        )
        fig.add_trace(
            go.Scatter(x=x, y=self.log["yddd"][:, 0], name="yddd"), row=4, col=1
        )
        fig.add_trace(
            go.Scatter(x=x, y=self.log["yddd"][:, 1], name="yddd"), row=4, col=1
        )
        fig.add_trace(
            go.Scatter(x=x, y=self.log["yddd"][:, 2], name="yddd"), row=4, col=1
        )
        fig.add_trace(go.Scatter(x=x, y=self.log["f"][:, 0], name="f"), row=5, col=1)
        fig.add_trace(go.Scatter(x=x, y=self.log["f"][:, 1], name="f"), row=5, col=1)
        fig.add_trace(go.Scatter(x=x, y=self.log["f"][:, 2], name="f"), row=5, col=1)
        fig.add_trace(go.Scatter(x=x, y=self.log["fd"][:, 0], name="fd"), row=6, col=1)
        fig.add_trace(go.Scatter(x=x, y=self.log["fd"][:, 1], name="fd"), row=6, col=1)
        fig.add_trace(go.Scatter(x=x, y=self.log["fd"][:, 2], name="fd"), row=6, col=1)
        fig.add_trace(
            go.Scatter(x=x, y=self.log["f_hat"][:, 0], name="f_hat"), row=5, col=1
        )
        fig.add_trace(
            go.Scatter(x=x, y=self.log["f_hat"][:, 1], name="f_hat"), row=5, col=1
        )
        fig.add_trace(
            go.Scatter(x=x, y=self.log["f_hat"][:, 2], name="f_hat"), row=5, col=1
        )
        fig.add_trace(
            go.Scatter(x=x, y=self.log["fd_hat"][:, 0], name="fd_hat"), row=6, col=1
        )
        fig.add_trace(
            go.Scatter(x=x, y=self.log["fd_hat"][:, 1], name="fd_hat"), row=6, col=1
        )
        fig.add_trace(
            go.Scatter(x=x, y=self.log["fd_hat"][:, 2], name="fd_hat"), row=6, col=1
        )
        fig.add_trace(
            go.Scatter(x=x, y=self.log["kd_hat"][:, 0], name="kd_hat"), row=7, col=1
        )
        fig.add_trace(
            go.Scatter(x=x, y=self.log["kd_hat"][:, 1], name="kd_hat"), row=7, col=1
        )
        fig.add_trace(
            go.Scatter(x=x, y=self.log["kd_hat"][:, 2], name="kd_hat"), row=7, col=1
        )
        fig.add_trace(
            go.Scatter(x=x, y=self.log["k_hat"][:, 0], name="k_hat"), row=8, col=1
        )
        fig.add_trace(
            go.Scatter(x=x, y=self.log["k_hat"][:, 1], name="k_hat"), row=8, col=1
        )
        fig.add_trace(
            go.Scatter(x=x, y=self.log["k_hat"][:, 2], name="k_hat"), row=8, col=1
        )

        fig.update_layout(height=1900, width=1200, title_text=title)
        fig.update_layout(hovermode="x unified")

        # fig.show()
        return fig


class VirtualTargetEstimator:
    def __init__(
        self,
        k_max,
        k_min,
        f_low,
        f_high,
        dim=3,
        characteristic_length=0.1,
    ):
        """
        k_max: maximum stiffness
        k_min: minimum stiffness
        f_low: lower bound of the force
        f_high: upper bound of the force
        dim: 3 or 6, 3 for translational, 6 for full 6D
        characteristic_length: the characteristic length for rotational stiffness
        """

        self.k_max = k_max
        self.k_min = k_min
        self.f_low = f_low
        self.f_high = f_high
        self.dim = dim
        self.characteristic_length = characteristic_length

        # internal variables
        self.last_wrench_T = np.zeros(6)
        self.last_pose_WT = np.array([0, 0, 0, 1, 0, 0, 0])

        assert self.dim in [3, 6]
        print("---------------------new PE --------------------")

    def update(self, wrench_T):
        """
        wrench_T: 6x1, wrench in the tool frame
        """
        # reg: convert rotational units to translational units
        # The goal is to use the same stiffness coefficient.
        # Real to Reg for characteristic length = 0.1m:
        #   angle: rad -> rad * m = m,  10 rad -> 1 m
        #   torque: Nm -> Nm / m = N,   10 Nm -> 100 N
        #   rotational stiffness: Nm/rad -> Nm/rad / m^2 = N / m, 1Nm/rad -> 100 N/m
        f = -wrench_T[: self.dim]
        f_reg = f.copy()
        if self.dim == 6:
            f_reg[3:] = f_reg[3:] / self.characteristic_length

        # compute stiffness
        f_norm = np.linalg.norm(f_reg)
        if f_norm < self.f_low:
            k = self.k_max
            twist_reg_TC = np.zeros(self.dim)
        elif f_norm > self.f_high:
            k = self.k_min
            twist_reg_TC = f_reg / k
        else:
            k = self.k_max - (self.k_max - self.k_min) * (f_norm - self.f_low) / (
                self.f_high - self.f_low
            )
            twist_reg_TC = f_reg / k

        twist_TC = twist_reg_TC

        if self.dim == 6:
            twist_TC[3:] = twist_TC[3:] / self.characteristic_length
            SE3_TC = su.spt_to_SE3(twist_TC)
            return k, SE3_TC
        else:
            pos_TC = twist_TC
            return k, pos_TC

    def batch_update(self, wrench_T):
        """
        wrench_T: Nx6, a batch of wrench in the tool frame
        """
        # reg: convert rotational units to translational units
        # The goal is to use the same stiffness coefficient.
        # Real to Reg for characteristic length = 0.1m:
        #   angle: rad -> rad * m = m,  10 rad -> 1 m
        #   torque: Nm -> Nm / m = N,   10 Nm -> 100 N
        #   rotational stiffness: Nm/rad -> Nm/rad / m^2 = N / m, 1Nm/rad -> 100 N/m
        f = -wrench_T[..., : self.dim]
        f_reg = f.copy()
        if self.dim == 6:
            f_reg[..., 3:] = f_reg[..., 3:] / self.characteristic_length

        # compute stiffness
        f_norm = np.linalg.norm(f_reg, axis=-1)
        k = np.zeros(f_norm.shape)[:, np.newaxis]
        twist_reg_TC = np.zeros(f.shape)
        mask = f_norm < self.f_low
        k[mask] = self.k_max
        twist_reg_TC[mask, :] = 0
        mask = f_norm > self.f_high
        k[mask] = self.k_min
        twist_reg_TC[mask] = f_reg[mask] / k[mask]
        mask = np.logical_and(f_norm >= self.f_low, f_norm <= self.f_high)
        k[mask] = self.k_max - (self.k_max - self.k_min) * (
            f_norm[mask][:, np.newaxis] - self.f_low
        ) / (self.f_high - self.f_low)
        twist_reg_TC[mask] = f_reg[mask] / k[mask]

        twist_TC = twist_reg_TC

        if self.dim == 6:
            twist_TC[..., 3:] = twist_TC[..., 3:] / self.characteristic_length
            SE3_TC = su.spt_to_SE3(twist_TC)
            return k, SE3_TC
        else:
            pos_TC = twist_TC
            return k, pos_TC

if __name__ == "__main__":
    import zarr
    from spatialmath import SE3
    from spatialmath.base import q2r

    dataset_path = "/home/yifanhou/data/real_processed/flip_up_102_0710"
    buffer = zarr.open(dataset_path, mode="r")

    for ep, ep_data in buffer["data"].items():
        print("Processing episode: ", ep)
        ts_pose_fb = ep_data["ts_pose_fb"]
        wrench_filtered = ep_data["wrench_filtered"]
        low_dim_time_stamps = ep_data["low_dim_time_stamps"]

        num_time_steps = len(low_dim_time_stamps)

        obs = StiffnessObserver(
            dt=0.01,
            alpha=1.0,
            mass=0.1,
            damping=1.8,
            k0=0.0,  # initial value of the stiffness
            yd_threshold=0.001,
            buffer_size=5,
        )

        for t in range(0, num_time_steps, 5):
            pose7_WT = ts_pose_fb[t]
            SE3_WT = SE3.Rt(q2r(pose7_WT[3:7]), pose7_WT[0:3], check=False)
            wrench_T = wrench_filtered[t]
            obs.update(SE3_WT.data[0], wrench_T)

        fig1 = obs.plot_log()
        fig1.show()

        input("Press Enter to continue...")
