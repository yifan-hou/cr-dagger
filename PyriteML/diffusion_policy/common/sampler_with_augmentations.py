import sys
import os

SCRIPT_PATH = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(SCRIPT_PATH, "../../../../"))

from typing import Optional, Callable
import numpy as np
from tqdm import tqdm
import cvxpy as cp
import copy

from plotly.subplots import make_subplots
import plotly.graph_objs as go

import PyriteUtility.spatial_math.spatial_utilities as su

from diffusion_policy.common.sampler import SequenceSampler


def compute_variation_score(array):
    minvalue = np.min(array, axis=0)
    maxvalue = np.max(array, axis=0)
    area_above_min = np.sum(array - minvalue, axis=0)
    area_below_max = np.sum(maxvalue - array, axis=0)
    return np.minimum(area_above_min, area_below_max)


def fit_polynomial_a1a2b1b2(wrench_proj, twist_proj, dN=1, method="least_square"):
    assert dN >= 1
    T = wrench_proj.shape[0]
    #         0 1 2 3 4 5 6 7 8 9
    # dN = 1: * * |
    # dN = 2: *   *   |
    # dN = 3: *     *     |
    Nspare = 2 * dN
    # solve for a1, a2, b0, b1, b2
    A = np.zeros((T - Nspare, 4))
    b = np.zeros((T - Nspare, 1))
    for row in np.arange(Nspare, T):
        A[row - Nspare, 0] = twist_proj[row - dN]
        A[row - Nspare, 1] = twist_proj[row - 2 * dN]
        A[row - Nspare, 2] = wrench_proj[row - dN]
        A[row - Nspare, 3] = 1
        b[row - Nspare] = -twist_proj[row]
    if method == "least_square":
        x = np.linalg.lstsq(A, b, rcond=None)[0]
    elif method == "qp":
        # option 2: constrained qp
        x = cp.Variable(4)
        P = A.T @ A
        q = -A.T @ b
        G = np.array(
            [
                [1, 0, 0, 0],
                [-1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, -1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, -1, 0],
                [0, 0, 0, 1],
                [0, 0, 0, -1],
            ]
        )
        # h = np.array([-1.972, 2.01, 1, -0.9861, -1.97e-7, 4e-6, 4e-4, 4e-4])
        h = np.array([-1, 3, 1, -0.5, -1e-7, 4e-5, 4e-3, 4e-3])
        prob = cp.Problem(
            cp.Minimize((1 / 2) * cp.quad_form(x, P) + q.T @ x), [G @ x <= h]
        )
        prob.solve()
        x = x.value
    a1 = x[0]
    a2 = x[1]
    b1 = x[2]
    b2 = x[3]
    return a1, a2, b1, b2


def augment_a1a2b1b2(a1, a2, b1, b2, wrench_proj, twist_proj, wrench_augs, dN=1):
    H = len(wrench_augs)
    T = wrench_proj.shape[0]

    wrench_proj_augs = np.zeros((H, T))
    twist_proj_augs = np.zeros((H, T))

    for i in range(H):
        delta_wrench_proj = wrench_augs[i] * np.ones_like(wrench_proj)
        delta_wrench_proj[:dN] = 0  # first dN step is not accounted for, must be zero
        wrench_proj_aug = wrench_proj + delta_wrench_proj
        twist_proj_aug = twist_proj.copy()

        for row in range(2 * dN, T):
            twist_proj_aug[row] = (
                -a1 * twist_proj_aug[row - dN]
                - a2 * twist_proj_aug[row - 2 * dN]
                - b1 * wrench_proj_aug[row - dN]
                - b2
            )
        wrench_proj_augs[i] = wrench_proj_aug
        twist_proj_augs[i] = twist_proj_aug

    return wrench_proj_augs, twist_proj_augs


# x_t = a1*x_{t-1} + a2*x_{t-2} + ... + a_k*x_{t-k}
#       + b1*u_{t-1} + b2*u_{t-2} + ... + b_{k-1}*u_{t-k+1} + b_k
def fit_polynomial_k(wrench_proj, twist_proj, k=2):
    assert k >= 2
    T = wrench_proj.shape[0]
    # solve for a1, a2, ..., ak; b1, b2, ..., bk
    A = np.zeros((T - k, 2 * k))
    b = np.zeros((T - k, 1))
    for row in np.arange(k, T):
        for i in range(k):
            A[row - k, i] = twist_proj[row - 1 - i]
            A[row - k, i + k] = wrench_proj[row - 1 - i]
        A[row - k, 2 * k - 1] = 1
        b[row - k] = twist_proj[row]
    x = np.linalg.lstsq(A, b, rcond=None)[0]
    return x


def augment_k(x, k, wrench_proj, twist_proj, wrench_augs):
    H = len(wrench_augs)
    T = wrench_proj.shape[0]

    wrench_proj_augs = np.zeros((H, T))
    twist_proj_augs = np.zeros((H, T))

    for i in range(H):
        delta_wrench_proj = wrench_augs[i] * np.ones_like(wrench_proj)
        delta_wrench_proj[0] = 0  # first force is not accounted for, must be zero
        wrench_proj_aug = wrench_proj + delta_wrench_proj
        twist_proj_aug = twist_proj.copy()

        for row in range(k, T):
            twist_proj_aug[row] = 0
            for j in range(k):
                twist_proj_aug[row] += x[j] * twist_proj_aug[row - 1 - j]
            for j in range(k - 1):
                twist_proj_aug[row] += x[j + k] * wrench_proj_aug[row - 1 - j]
            twist_proj_aug[row] += x[2 * k - 1]

        wrench_proj_augs[i] = wrench_proj_aug
        twist_proj_augs[i] = twist_proj_aug

    return wrench_proj_augs, twist_proj_augs


def augment_sample_qp(sampler, augment_config):
    # augment_config = {
    #     "nominal_length_m": 0.1, used to trade off between translation and rotation
    #     "wrench_variation_score_threshold": 10, 1.5N * h / 2
    #     "aug_max_force": 10,
    #     "dim": 6,
    #     "visualize": False,
    # }
    # returns
    #   sample_id: id for the original sample that the augmentation is based on
    #   obs_dict: augmented observation dictionary, only includes low dim keys. rgb keys should be queried from the original sampler using sample_id
    #   action_array: augmented action array
    print("Augmenting samples.")
    print(f"Augmenting with dim: {augment_config['dim']}")
    print(
        f"Augmenting with wrench_variation_score_threshold: {augment_config['wrench_variation_score_threshold']}"
    )
    print(f"Augmenting with aug_max_force: {augment_config['aug_max_force']}")
    print(f"Augmenting with visualize: {augment_config['visualize']}")

    sample_ids = []
    obs_augments = []
    action_augments = []

    dim = augment_config["dim"]

    for id in tqdm(range(sampler.__len__())):
        obs, action = sampler.sample_sequence(id)
        obs_wrench = obs["dense"]["robot0_eef_wrench"]  # H, T, D
        obs_pos = obs["dense"]["robot0_eef_pos"]
        obs_rot6 = obs["dense"]["robot0_eef_rot_axis_angle"]
        obs_pose9 = np.concatenate([obs_pos, obs_rot6], axis=-1)  # H, T, 9
        obs_SE3 = su.pose9_to_SE3(obs_pose9)  # H, T, 4, 4

        action_pos9 = action["dense"][:, :-1, :]  # H, T-1, 9
        action_SE3 = su.pose9_to_SE3(action_pos9)  # H, T-1, 4, 4
        action_wrench = action["dense_wrench"][:, :-1, :]  # H, T-1, 6

        Tobs = obs_wrench.shape[1]
        Taction = action_wrench.shape[1]

        comb_SE3 = np.concatenate([obs_SE3, action_SE3], axis=1)  # H, T', 4, 4
        comb_wrench = np.concatenate([obs_wrench, action_wrench], axis=1)  # H, T', D

        assert comb_SE3.shape[0] == comb_wrench.shape[0]
        assert comb_SE3.shape[1] == comb_wrench.shape[1]
        H, T, _, _ = comb_SE3.shape

        # print(f"H = {H}, T = {T}, Tobs = {Tobs}, Taction = {Taction}")
        # Check if augmentation should be added
        best_variation_score = 0
        best_proj_vector = np.zeros((dim))
        best_wrench_proj = np.zeros((T))
        best_twist_proj = np.zeros((T))
        best_wrench_piece = np.zeros((T, dim))
        best_twist_piece = np.zeros((T, dim))
        for h in range(H):
            # for each piece
            wrench_piece = comb_wrench[h, :, :dim]  # T', D
            # no need to compute relative pose since this was already done in the sampler
            SE3_piece = comb_SE3[h]  # T', 4, 4
            if dim == 3:
                twist_piece = SE3_piece[:, :3, 3]
            else:
                twist_piece = np.zeros((T, dim))
                for j in range(T):
                    twist_piece[j, :] = su.SE3_to_spt(SE3_piece[j, :], 1e-15)

            # Find a large variation direction
            wrench_piece_delta = np.diff(wrench_piece, axis=0)
            max_id = np.argmax(np.linalg.norm(wrench_piece_delta, axis=1))
            vector = wrench_piece[0] - wrench_piece[max_id + 1]
            vector = vector / np.linalg.norm(vector)
            # project wrench and pose to the variation direction
            wrench_proj = np.sum(wrench_piece * vector, axis=1)
            twist_proj = np.sum(twist_piece * vector, axis=1)
            # compute the variation score
            variation_score = compute_variation_score(wrench_proj)
            if variation_score > best_variation_score:
                best_variation_score = variation_score
                best_proj_vector = vector
                best_wrench_proj = wrench_proj
                best_twist_proj = twist_proj
                best_wrench_piece = wrench_piece
                best_twist_piece = twist_piece

        if best_variation_score < augment_config["wrench_variation_score_threshold"]:
            print(
                f"Skipping sample {id} due to low variation score: {best_variation_score}."
            )
            continue
        ##
        ## Augment the observation
        ##
        a1, a2, b1, b2 = fit_polynomial_a1a2b1b2(best_wrench_proj, best_twist_proj)

        ## Generate H augmentations
        wrench_augs = np.arange(1, H + 1) / H * augment_config["aug_max_force"]
        wrench_proj_augs, twist_proj_augs = augment_a1a2b1b2(
            a1,
            a2,
            b1,
            b2,
            best_wrench_proj,
            best_twist_proj,
            wrench_augs,
        )

        wrench_proj_delta_all = wrench_proj_augs - best_wrench_proj
        twist_proj_delta_all = twist_proj_augs - best_twist_proj

        ## illustrate the result
        if augment_config["visualize"]:
            print(f"a1:{a1}, a2:{a2}, b1:{b1}, b2:{b2}.")
            c = input("Press Enter to continue, d + Enter to draw")
            if c == "d":
                # fmt: off
                fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True, subplot_titles=('pos', 'force'))
                fig.add_trace(go.Scatter(x=np.arange(T), y=best_twist_proj, name='best_twist_proj', mode="markers",),row=1, col=1)
                for f in range(H):
                    fig.add_trace(go.Scatter(x=np.arange(T), y=twist_proj_augs[f], name=f'twist_proj_aug_{f}', mode="markers",),row=1, col=1)
                fig.add_trace(go.Scatter(x=np.arange(T), y=best_wrench_proj, name='best_wrench_proj', mode="markers",),row=2, col=1)
                for f in range(H):
                    fig.add_trace(go.Scatter(x=np.arange(T), y=wrench_proj_delta_all[f], name=f'wrench_proj_aug_{f}', mode="markers"),row=2, col=1)
                fig.update_layout(height=600, width=800)
                fig.show()
                # fmt: on
                print("Done for one")
        ##
        ## Recover the augmented pose/wrenches from the projection
        ##
        # HxT * 6x1 = HxTx6
        wrench_delta = wrench_proj_delta_all[..., np.newaxis] * best_proj_vector
        twist_delta = twist_proj_delta_all[..., np.newaxis] * best_proj_vector
        # Tx6 + HxTx6 = HxTx6
        wrench_augments = best_wrench_piece + wrench_delta
        twist_augments = best_twist_piece + twist_delta
        pose9_augments = su.SE3_to_pose9(su.spt_to_SE3(twist_augments, 1e-15))
        pos_augments = pose9_augments[..., :3]
        rot6_augments = pose9_augments[..., 3:]

        sample_ids.append(id)
        obs_augments.append(
            {
                "dense": {
                    "robot0_eef_wrench": wrench_augments[:, :Tobs, :].astype(
                        np.float32
                    ),
                    "robot0_eef_pos": pos_augments[:, :Tobs, :].astype(np.float32),
                    "robot0_eef_rot_axis_angle": rot6_augments[:, :Tobs, :].astype(
                        np.float32
                    ),
                }
            }
        )
        action_augments.append(
            {
                "dense": pose9_augments[:, Tobs - 1 :, :].astype(np.float32),
                "dense_wrench": action["dense_wrench"],
            }
        )

        # debug
        if obs_augments[-1]["dense"]["robot0_eef_pos"].shape[0] != 10:
            print("---------- Error ----------")
            print("pos shape: ", obs_augments[-1]["dense"]["robot0_eef_pos"].shape)
            print(
                "wrench shape: ", obs_augments[-1]["dense"]["robot0_eef_wrench"].shape
            )
            print("H: ", H)
            print("Tobs: ", Tobs)
            print("Taction: ", Taction)
            print("obs_wrench.shape: ", obs_wrench.shape)
            print("action_wrench.shape: ", action_wrench.shape)
            print("comb_wrench.shape: ", comb_wrench.shape)
            print("comb_SE3.shape: ", comb_SE3.shape)
            print("wrench_proj_augs: ", wrench_proj_augs)

    return sample_ids, obs_augments, action_augments


class SequenceSamplerWithAugmentSamples(SequenceSampler):
    def __init__(
        self,
        shape_meta: dict,
        replay_buffer: dict,
        obs_to_obs_sample: Callable,
        action_to_action_sample: Callable,
        id_list: list,
        sparse_query_frequency_down_sample_steps: int = 1,
        episode_mask: Optional[np.ndarray] = None,
        action_padding: bool = False,
        augment_config: dict = {},
    ):
        super().__init__(
            shape_meta=shape_meta,
            replay_buffer=replay_buffer,
            obs_to_obs_sample=obs_to_obs_sample,
            action_to_action_sample=action_to_action_sample,
            id_list=id_list,
            sparse_query_frequency_down_sample_steps=sparse_query_frequency_down_sample_steps,
            episode_mask=episode_mask,
            action_padding=action_padding,
        )

        aug_sample_ids, aug_obs, aug_action = augment_sample_qp(super(), augment_config)

        print(
            f"Original data has {super().__len__()} samples. Augmented {len(aug_sample_ids)} samples."
        )
        self.aug_sample_ids = aug_sample_ids
        self.aug_obs = aug_obs
        self.aug_action = aug_action

    def __len__(self):
        return super().__len__() + len(self.aug_sample_ids)

    def sample_sequence(self, idx):
        if idx < super().__len__():
            return super().sample_sequence(idx)
        else:
            idx = idx - super().__len__()
            obs, action = super().sample_sequence(self.aug_sample_ids[idx])
            obs_new = self.aug_obs[idx]
            action_new = self.aug_action[idx]

            obs["dense"] = copy.deepcopy(obs_new["dense"])
            action["dense"] = copy.deepcopy(action_new["dense"])
            action["dense_wrench"] = copy.deepcopy(action_new["dense_wrench"])
            return obs, action
