import sys
import os

SCRIPT_PATH = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(SCRIPT_PATH, "../../../"))

from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.model.diffusion.mlp import MLP
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy.model.vision.timm_obs_encoder_with_force import (
    TimmObsEncoderWithForce,
)
from diffusion_policy.common.pytorch_util import dict_apply

from PyriteUtility.data_pipeline.indexing import get_dense_query_points_in_horizon
from PyriteUtility.data_pipeline.data_plotting import plot_ts_action
from PyriteUtility.planning_control.trajectory import LinearInterpolator


class TransformerMLPPolicy(BaseImagePolicy):
    def __init__(
        self,
        shape_meta: dict,
        obs_encoder: TimmObsEncoderWithForce,
        mlp_only: bool = False,
        **kwargs,
    ):
        super().__init__()

        # parse shapes
        action_shape = shape_meta["action"]["shape"]
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        action_horizon = shape_meta["sample"]["action"]["dense"]["horizon"]
        action_down_sample_steps = shape_meta["sample"]["action"]["dense"][
            "down_sample_steps"
        ]
        # get feature dim
        obs_feature_dim = np.prod(obs_encoder.output_shape())
        if mlp_only:
            obs_feature_dim = 0

        # compute input dimension of the dense model
        # input_dim = dense_obs_dim + obs_feature_dim (only if mlp_only is False)
        # where:
        #  dense_obs_dim = sum([dense_obs_shape[key]*dense_obs_horizon[key] for each key])
        dense_input_dim = 0
        for key, attr in shape_meta["sample"]["obs"]["dense"].items():
            horizon = attr["horizon"]
            assert (
                len(shape_meta["obs"][key]["shape"]) == 1
            )  # assuming dense obs shape is 1D
            size = shape_meta["obs"][key]["shape"][0]
            dense_input_dim += size * (
                horizon + 1
            )  # +1 because of get_dense_query_points_in_horizon() gives 6 points for horizon=5
            print(
                f"[TransformerMLPPolicy] key: {key}, size: {size}, horizon: {horizon}"
            )
        dense_input_dim += obs_feature_dim
        print(f"dense_input_dim: {dense_input_dim}")
        print(f"obs_feature_dim: {obs_feature_dim}")

        dense_output_dim = (action_horizon + 1) * action_dim
        model_dense = MLP(
            in_channels=dense_input_dim,
            out_channels=dense_output_dim,  # (B, T*D)
            hidden_channels=[256, 256],
        )

        self.obs_encoder = obs_encoder
        self.model_dense = model_dense
        self.sparse_normalizer = LinearNormalizer()
        self.dense_normalizer = LinearNormalizer()
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.action_down_sample_steps = action_down_sample_steps
        self.mlp_only = mlp_only
        self.kwargs = kwargs

        self.loss = 0

    def set_normalizer(
        self,
        sparse_normalizer: LinearNormalizer,
        dense_normalizer: LinearNormalizer = None,
    ):
        self.sparse_normalizer.load_state_dict(sparse_normalizer.state_dict())
        assert dense_normalizer is not None
        self.dense_normalizer.load_state_dict(dense_normalizer.state_dict())

    def get_normalizer(self):
        return self.sparse_normalizer, self.dense_normalizer

    def run_encoder(
        self,
        obs: Dict,
    ) -> Dict[str, torch.Tensor]:
        """
        obs: include keys from shape_meta['sample']['obs'],
            which should be a dictionary with keys 'sparse' and optionally 'dense'
        """
        if self.mlp_only:
            return None
        nobs_sparse = self.sparse_normalizer.normalize(obs["sparse"])
        return self.obs_encoder(nobs_sparse)  # this is sparse_nobs_encode

    def run_mlp(
        self,
        dense_obs_dict: Dict[
            str, torch.Tensor
        ],  # (B, H, T, D) or (B, H, T, D1, D2), H might = 1
        sparse_nobs_encode: torch.Tensor,  # (B, D)
        unnormalize_result: bool = True,
    ) -> torch.Tensor:
        """
        dense_obs_dict: include keys from shape_meta['obs']['dense'].
            Each key is a tensor of shape (B, H, T, D) or (B, H, T, D1, D2)
        sparse_nobs_encode: obs_encoder outputs. (B, D)

        When unnormalize_result is true, this function is used for control. We must have B = H = 1.
        """
        nobs_dense = self.dense_normalizer.normalize(dense_obs_dict)

        # x: force and pose
        dense_features = []
        for key in nobs_dense.keys():
            data = nobs_dense[key]  # (B, H, T, D)
            dense_features.append(rearrange(data, "b h ... -> b h (...)"))

        # concatenate all dense_features
        X = torch.cat(dense_features, dim=-1)  # (B, H, TxD)

        if X.shape[1] == 1:
            # Inference for a single control.
            # Should only happen during testing
            assert X.shape[0] == 1
            x = rearrange(X, "1 1 d -> 1 d")

            if self.mlp_only:
                pred = self.model_dense(x)
            else:
                pred = self.model_dense(
                    torch.cat([x, sparse_nobs_encode], axis=-1)
                )  # (B, T*D)
            dense_predictions = rearrange(
                pred, "b (t d) -> b t d", t=self.action_horizon + 1
            )
        else:
            # Inference for a batch.
            # Should only happen during training
            if not self.mlp_only:
                cond = rearrange(sparse_nobs_encode, "b ... -> b (...)")
            dense_predictions = []
            for i in range(X.shape[1]):
                x = X[:, i, ...]
                # x = rearrange(X[:, i, ...], "b t d -> b (t d)")
                if self.mlp_only:
                    pred = self.model_dense(x)
                else:
                    pred = self.model_dense(torch.cat([x, cond], axis=-1))  # (B, T*D)
                pred = rearrange(pred, "b (t d) -> b t d", t=self.action_horizon + 1)
                dense_predictions.append(pred)
            dense_predictions = torch.stack(dense_predictions, dim=1)  # (B, H, T, D)

        # # debug: plot action
        # dense_action_timesteps_local = np.arange(self.action_horizon + 1) * self.dense_action_down_sample_steps
        # dense_action_timesteps_h = dense_action_timesteps_local + time

        # sparse_action_time = np.arange(self.sparse_action_horizon) * self.sparse_action_down_sample_steps
        # sparse_action = sparse_ntraj[0, ...].detach().cpu().numpy()
        # dense_action_time = dense_action_timesteps_h
        # dense_action = dense_predictions[0, ...].detach().cpu().numpy()
        # plot_ts_action(sparse_action_time, sparse_action, dense_action_time, dense_action)
        # print('press Enter to continue')
        # input()

        if unnormalize_result:
            dense_predictions = self.dense_normalizer["action"].unnormalize(
                dense_predictions
            )
        return dense_predictions

    # predict action for a batch of data
    def predict_action(self, batch_obs, unnormalize_result=True):
        # run inference
        sparse_nobs_encode = self.run_encoder(batch_obs)
        dense_naction_pred = self.run_mlp(
            dense_obs_dict=batch_obs["dense"],
            sparse_nobs_encode=sparse_nobs_encode,  # (B, D)
            unnormalize_result=unnormalize_result,
        )

        return dense_naction_pred

    def compute_loss(self, batch):
        assert "valid_mask" not in batch

        # run inference
        sparse_nobs_encode = self.run_encoder(batch["obs"])
        dense_naction_pred = self.run_mlp(
            dense_obs_dict=batch["obs"]["dense"],
            sparse_nobs_encode=sparse_nobs_encode,  # (B, D)
            unnormalize_result=False,
        )

        # compute loss
        dense_naction = self.dense_normalizer["action"].normalize(
            batch["action"]["dense"]
        )  # (B, H, T, D)

        loss = F.mse_loss(dense_naction_pred, dense_naction, reduction="none")
        loss = loss.type(loss.dtype)
        loss = reduce(loss, "b ... -> b (...)", "mean")
        loss = loss.mean()

        # # regularization loss
        # pred_sparse # (B, T, D)
        # pred_vel = pred_sparse[:, 1:, ...] - pred_sparse[:, :-1, ...]
        # pred_acc = pred_vel[:, 1:, ...] - pred_vel[:, :-1, ...]
        # pred_jrk = pred_acc[:, 1:, ...] - pred_acc[:, :-1, ...]

        # smoothness_loss = F.mse_loss(pred_vel, torch.zeros_like(pred_vel), reduction='mean')
        #                 # + F.mse_loss(pred_acc, torch.zeros_like(pred_acc), reduction='mean') \
        #                 # + F.mse_loss(pred_jrk, torch.zeros_like(pred_jrk), reduction='mean')
        # smoothness_loss *= args['normalization_weight']

        return loss

    def forward(self, batch):
        return self.compute_loss(batch)
