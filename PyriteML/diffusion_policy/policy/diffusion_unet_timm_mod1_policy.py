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
from diffusion_policy.model.diffusion.mlp import MLP_conditioned_with_time_encoding
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy.model.vision.timm_obs_encoder_with_force import (
    TimmObsEncoderWithForce,
)
from diffusion_policy.common.pytorch_util import dict_apply

from PyriteUtility.data_pipeline.indexing import get_dense_query_points_in_horizon
from PyriteUtility.data_pipeline.data_plotting import plot_ts_action
from PyriteUtility.planning_control.trajectory import LinearInterpolator


class DiffusionUnetTimmMod1Policy(BaseImagePolicy):
    def __init__(
        self,
        shape_meta: dict,
        noise_scheduler: DDPMScheduler,
        obs_encoder: TimmObsEncoderWithForce,
        num_inference_steps=None,
        diffusion_step_embed_dim=256,
        down_dims=(256, 512, 1024),
        kernel_size=5,
        n_groups=8,
        cond_predict_scale=True,
        input_pertub=0.1,
        inpaint_fixed_action_prefix=False,
        train_diffusion_n_samples=1,
        hack_no_obs_encoder_for_dense=False,
        # parameters passed to step
        **kwargs,
    ):
        super().__init__()

        # parse shapes
        action_shape = shape_meta["action"]["shape"]
        assert len(action_shape) == 1
        # assuming sparse and dense actions have the same shape
        action_dim = action_shape[0]
        sparse_action_horizon = shape_meta["sample"]["action"]["sparse"]["horizon"]
        sparse_action_down_sample_steps = shape_meta["sample"]["action"]["sparse"][
            "down_sample_steps"
        ]
        flag_has_dense = "dense" in shape_meta["sample"]["action"]
        if flag_has_dense:
            dense_action_horizon = shape_meta["sample"]["action"]["dense"]["horizon"]
            dense_action_down_sample_steps = shape_meta["sample"]["action"]["dense"][
                "down_sample_steps"
            ]
            dense_sample_delta_steps = shape_meta["sample"]["action"][
                "dense_sample_delta_steps"
            ]
        # get feature dim
        obs_feature_dim = np.prod(obs_encoder.output_shape())

        # create diffusion model
        input_dim = action_dim
        global_cond_dim = obs_feature_dim

        model_sparse = ConditionalUnet1D(
            input_dim=input_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale,
        )

        if flag_has_dense:
            # compute input dimension of the dense model
            # input_dim = dense_obs_dim + cond_dim
            # where:
            #  dense_obs_dim = sum([dense_obs_shape[key]*dense_obs_horizon[key] for each key])
            #  cond_dim = sparse_traj_output_dim + obs_encoder_output_dim
            #           = sparse_action_horizon * action_dim + obs_encoder.output_shape()
            dense_input_dim = 0
            for key, attr in shape_meta["sample"]["obs"]["dense"].items():
                horizon = attr["horizon"]
                assert (
                    len(shape_meta["obs"][key]["shape"]) == 1
                )  # assuming dense obs shape is 1D
                size = shape_meta["obs"][key]["shape"][0]
                dense_input_dim += size * horizon
            dense_input_dim += sparse_action_horizon * action_dim
            if not hack_no_obs_encoder_for_dense:
                dense_input_dim += obs_feature_dim

            dense_output_dim = (dense_action_horizon + 1) * action_dim
            model_dense = MLP_conditioned_with_time_encoding(
                in_channels=dense_input_dim,
                out_channels=dense_output_dim,  # (B, T*D)
                action_horizon=sparse_action_horizon * sparse_action_down_sample_steps,
            )

        self.obs_encoder = obs_encoder
        self.model_sparse = model_sparse
        self.noise_scheduler = noise_scheduler
        self.sparse_normalizer = LinearNormalizer()
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.sparse_action_horizon = sparse_action_horizon
        self.sparse_action_down_sample_steps = sparse_action_down_sample_steps
        self.input_pertub = input_pertub
        self.inpaint_fixed_action_prefix = inpaint_fixed_action_prefix
        self.train_diffusion_n_samples = int(train_diffusion_n_samples)
        self.kwargs = kwargs
        self.sparse_loss = 0

        self.flag_has_dense = flag_has_dense
        self.hack_no_obs_encoder_for_dense = hack_no_obs_encoder_for_dense
        self.dense_normalizer = None
        self.dense_loss = 0
        if flag_has_dense:
            self.model_dense = model_dense
            self.dense_normalizer = LinearNormalizer()
            self.dense_action_horizon = dense_action_horizon
            self.dense_action_down_sample_steps = dense_action_down_sample_steps
            self.dense_sample_delta_steps = dense_sample_delta_steps

        # store intermediate results from sparse model
        self.sparse_nobs_encode = None
        self.sparse_naction_pred = None

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

    # ========= training  ============
    def set_normalizer(
        self,
        sparse_normalizer: LinearNormalizer,
        dense_normalizer: LinearNormalizer = None,
    ):
        self.sparse_normalizer.load_state_dict(sparse_normalizer.state_dict())
        if self.flag_has_dense:
            assert dense_normalizer is not None
            self.dense_normalizer.load_state_dict(dense_normalizer.state_dict())

    def get_normalizer(self):
        return self.sparse_normalizer, self.dense_normalizer

    # ========= inference  ============
    def conditional_sample(
        self,
        condition_data,
        condition_mask,
        local_cond=None,
        global_cond=None,
        generator=None,
        # keyword arguments to scheduler.step
        **kwargs,
    ):
        model = self.model_sparse
        scheduler = self.noise_scheduler

        trajectory = torch.randn(
            size=condition_data.shape,
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator,
        )

        # set step values
        scheduler.set_timesteps(self.num_inference_steps)

        for t in scheduler.timesteps:
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]

            # 2. predict model output
            model_output = model(
                trajectory, t, local_cond=local_cond, global_cond=global_cond
            )

            # 3. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(
                model_output, t, trajectory, generator=generator, **kwargs
            ).prev_sample

        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]

        return trajectory

    def predict_dense_action(
        self,
        dense_obs_dict: Dict[
            str, torch.Tensor
        ],  # (B, H, T, D) or (B, H, T, D1, D2), H might = 1
        time: torch.Tensor = None,  # (B, T)
        sparse_nobs_encode: torch.Tensor = None,  # (B, D)
        sparse_ntraj: torch.Tensor = None,  # (B, T, D)
        unnormalize_result: bool = True,
    ) -> torch.Tensor:
        """
        dense_obs_dict: include keys from shape_meta['obs']['dense'].
            Each key is a tensor of shape (B, H, T, D) or (B, H, T, D1, D2)
        time: time index. Range of value is [0, sparse_action_horizon*sparse_action_down_sample_steps]. (B, T)
        sparse_nobs_encode: obs_encoder outputs. (B, D)
        sparse_ntraj: sparse action outputs. Still normalized. (B, T, D)

        When time is not None, this function is used for control. We must have B = H = 1.
        """
        nobs_dense = self.dense_normalizer.normalize(dense_obs_dict)
        if sparse_nobs_encode is None:
            sparse_nobs_encode = self.sparse_nobs_encode
        if sparse_ntraj is None:
            sparse_ntraj = self.sparse_naction_pred
        assert sparse_nobs_encode is not None
        assert sparse_ntraj is not None

        # x: force and pose
        dense_features = []
        for key in nobs_dense.keys():
            data = nobs_dense[key]  # (B, H, T, D) or (B, H, T, D1, D2)
            dense_features.append(rearrange(data, "b h t ... -> b h t (...)"))

        # concatenate all dense_features
        X = torch.cat(dense_features, dim=-1)  # (B, H, T, D)

        # cond: sparse traj and obs encode
        if self.hack_no_obs_encoder_for_dense:
            cond = rearrange(sparse_ntraj, "b t d ... -> b (t d ...)")
        else:
            cond = torch.cat(
                [
                    rearrange(sparse_ntraj, "b t d ... -> b (t d ...)"),
                    rearrange(sparse_nobs_encode, "b ... -> b (...)"),
                ],
                axis=-1,
            )

        if time is not None:
            # control for one timestep. B = H = 1
            assert X.shape[1] == 1
            x = rearrange(X, "1 1 t d -> 1 (t d)")
            t = torch.tensor([time], dtype=torch.long, device=x.device)
            pred = self.model_dense(x, cond, t)  # (B, T*D)
            dense_predictions = rearrange(
                pred, "b (t d) -> b t d", t=self.dense_action_horizon + 1
            )
        else:
            # Inference for a batch.
            dense_predictions = []
            time_steps = get_dense_query_points_in_horizon(
                self.sparse_action_horizon * self.sparse_action_down_sample_steps,
                self.dense_action_horizon,
                self.dense_action_down_sample_steps,
                self.dense_sample_delta_steps,
            )
            for i in range(X.shape[1]):
                x = rearrange(X[:, i, ...], "b t d -> b (t d)")
                t = torch.tensor([time_steps[i]], dtype=torch.long, device=x.device)
                pred = self.model_dense(x, cond, t)  # (B, T*D)
                pred = rearrange(
                    pred, "b (t d) -> b t d", t=self.dense_action_horizon + 1
                )
                dense_predictions.append(pred)
            dense_predictions = torch.stack(dense_predictions, dim=1)  # (B, H, T, D)

        # # debug: plot action
        # dense_action_timesteps_local = np.arange(self.dense_action_horizon + 1) * self.dense_action_down_sample_steps
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

    # def predict_dense_action(self,
    #     dense_obs_dict: Dict[str, torch.Tensor], # (B, H, T, D) or (B, H, T, D1, D2), H might = 1
    #     time: torch.Tensor = None, # (B, T)
    #     sparse_nobs_encode: torch.Tensor = None, # (B, D)
    #     sparse_ntraj: torch.Tensor = None, # (B, T, D)
    #     unnormalize_result: bool = True
    # ) -> torch.Tensor:
    #     """
    #     hacked version, replace dense network with linear interpolation
    #     """
    #     if sparse_ntraj is None:
    #         sparse_ntraj = self.sparse_naction_pred # (B, T, D)

    #     sparse_data_time = np.arange(self.sparse_action_horizon) * self.sparse_action_down_sample_steps
    #     sparse_trajs = []
    #     for i in range(sparse_ntraj.shape[0]):
    #         sparse_trajs.append(LinearInterpolator(sparse_data_time, sparse_ntraj[i, ...].detach().cpu().numpy()))

    #     dense_action_timesteps_local = np.arange(self.dense_action_horizon + 1) * self.dense_action_down_sample_steps

    #     if time is not None:
    #         # control for one timestep. B = H = 1
    #         dense_action_timesteps_h = dense_action_timesteps_local + time
    #         pred = sparse_trajs[0](dense_action_timesteps_h)
    #         dense_predictions = rearrange(pred, 't d -> 1 t d')
    #     else:
    #         # Inference for a batch.
    #         dense_predictions = []
    #         time_steps = get_dense_query_points_in_horizon(self.sparse_action_horizon * self.sparse_action_down_sample_steps,
    #                                                        self.dense_action_horizon,
    #                                                        self.dense_action_down_sample_steps,
    #                                                        self.dense_sample_delta_steps)
    #         for i in range(X.shape[1]):
    #             x = rearrange(X[:, i, ...], 'b t d -> b (t d)')
    #             t = torch.tensor([time_steps[i]], dtype=torch.long, device=x.device)
    #             pred = self.model_dense(x, cond, t) # (B, T*D)
    #             pred = rearrange(pred, 'b (t d) -> b t d', t=self.dense_action_horizon+1)
    #             dense_predictions.append(pred)
    #         dense_predictions = torch.stack(dense_predictions, dim=1) # (B, H, T, D)

    #     # # debug: plot action
    #     # sparse_action_time = np.arange(self.sparse_action_horizon) * self.sparse_action_down_sample_steps
    #     # sparse_action = sparse_ntraj[0, ...].detach().cpu().numpy()
    #     # dense_action_time = dense_action_timesteps_h
    #     # dense_action = dense_predictions[0, ...]
    #     # plot_ts_action(sparse_action_time, sparse_action, dense_action_time, dense_action, title = 'in the policy')
    #     # print('press Enter to continue')
    #     # input()

    #     if unnormalize_result:
    #         # use sparse normalizer, since the result is an interpolation of sparse actions
    #         dense_predictions = self.sparse_normalizer['action'].unnormalize(dense_predictions)
    #     return dense_predictions

    def predict_action(
        self,
        obs: Dict,
        debug_action: Dict = None,
    ) -> Dict[str, torch.Tensor]:
        """
        obs: include keys from shape_meta['sample']['obs'],
            which should be a dictionary with keys 'sparse' and optionally 'dense'
        debug_action: if provided, gt action will be used for sparse_ntraj in dense prediction
        """
        obs_dict_sparse = obs["sparse"]

        ##
        ## =================  Part one: Sparse =================
        ##
        nobs_sparse = self.sparse_normalizer.normalize(obs_dict_sparse)

        batch_size = next(iter(nobs_sparse.values())).shape[0]

        # condition through global feature
        sparse_nobs_encode = self.obs_encoder(nobs_sparse)

        # empty data for action
        cond_data = torch.zeros(
            size=(batch_size, self.sparse_action_horizon, self.action_dim),
            device=self.device,
            dtype=self.dtype,
        )
        cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)

        # run sampling
        sparse_naction_pred = self.conditional_sample(
            condition_data=cond_data,
            condition_mask=cond_mask,
            local_cond=None,
            global_cond=sparse_nobs_encode,
            **self.kwargs,
        )

        # unnormalize prediction
        assert sparse_naction_pred.shape == (
            batch_size,
            self.sparse_action_horizon,
            self.action_dim,
        )
        sparse_action_pred = self.sparse_normalizer["action"].unnormalize(
            sparse_naction_pred
        )

        if debug_action is not None:
            sparse_naction_pred = self.sparse_normalizer["action"].normalize(
                debug_action["sparse"]
            )
        self.sparse_nobs_encode = sparse_nobs_encode
        self.sparse_naction_pred = sparse_naction_pred

        ##
        ## =================  Part two: Dense =================
        ##
        # obs_dict_dense: include keys from shape_meta['sample']['obs']['dense'], each key is a tensor of shape (B, H, T, D) or (B, H, T, D1, D2)
        dense_action_pred = None
        # self.flag_has_dense = False: the policy does not have dense part
        # 'dense' not in obs: this is the sparse inference of the hybrid policy, no dense part
        if self.flag_has_dense and "dense" in obs:
            obs_dict_dense = obs["dense"]
            dense_action_pred = self.predict_dense_action(
                dense_obs_dict=obs_dict_dense,
                sparse_nobs_encode=sparse_nobs_encode,
                sparse_ntraj=sparse_naction_pred,
                unnormalize_result=True,
            )

        result = {"sparse": sparse_action_pred, "dense": dense_action_pred}
        return result

    def compute_loss(self, batch, args):
        # normalize input
        assert "valid_mask" not in batch
        nobs_sparse = self.sparse_normalizer.normalize(batch["obs"]["sparse"])
        nactions_sparse = self.sparse_normalizer["action"].normalize(
            batch["action"]["sparse"]
        )

        sparse_nobs_encode = self.obs_encoder(nobs_sparse)

        ##
        ## =================  Part one: Sparse =================
        ##
        trajectory = nactions_sparse

        # Sample noise that we'll add to the images
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        # input perturbation by adding additonal noise to alleviate exposure bias
        # reference: https://github.com/forever208/DDPM-IP
        noise_new = noise + self.input_pertub * torch.randn(
            trajectory.shape, device=trajectory.device
        )

        # Sample a random timestep for each image
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (nactions_sparse.shape[0],),
            device=trajectory.device,
        ).long()

        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise_new, timesteps
        )

        # Predict the noise residual
        pred_sparse = self.model_sparse(
            noisy_trajectory, timesteps, local_cond=None, global_cond=sparse_nobs_encode
        )

        pred_type = self.noise_scheduler.config.prediction_type
        if pred_type == "epsilon":
            target = noise
        elif pred_type == "sample":
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        # # regularization loss
        # pred_sparse # (B, T, D)
        # pred_vel = pred_sparse[:, 1:, ...] - pred_sparse[:, :-1, ...]
        # pred_acc = pred_vel[:, 1:, ...] - pred_vel[:, :-1, ...]
        # pred_jrk = pred_acc[:, 1:, ...] - pred_acc[:, :-1, ...]

        # smoothness_loss = F.mse_loss(pred_vel, torch.zeros_like(pred_vel), reduction='mean')
        #                 # + F.mse_loss(pred_acc, torch.zeros_like(pred_acc), reduction='mean') \
        #                 # + F.mse_loss(pred_jrk, torch.zeros_like(pred_jrk), reduction='mean')
        # smoothness_loss *= args['normalization_weight']
        smoothness_loss = 0

        sparse_loss = F.mse_loss(pred_sparse, target, reduction="mean")
        self.sparse_loss = sparse_loss
        self.smoothness_loss = smoothness_loss

        loss = sparse_loss + smoothness_loss

        ##
        ## =================  Part two: Dense =================
        ##
        if self.flag_has_dense and args["start_training_dense"]:
            if args["dense_traj_cond_use_gt"]:
                sparse_ntraj_cond = nactions_sparse
            else:
                if pred_type == "epsilon":
                    pred_trajectory = noisy_trajectory - pred_sparse
                else:
                    pred_trajectory = pred_sparse
                sparse_ntraj_cond = pred_trajectory.detach()
            # sparse_nobs_encode_detached = sparse_nobs_encode.detach()
            dense_naction_pred = self.predict_dense_action(
                dense_obs_dict=batch["obs"]["dense"],
                sparse_nobs_encode=sparse_nobs_encode,  # (B, D)
                sparse_ntraj=sparse_ntraj_cond,  # (B, T, D)
                unnormalize_result=False,
            )

            # compute loss
            nactions_dense = self.dense_normalizer["action"].normalize(
                batch["action"]["dense"]
            )
            dense_naction = nactions_dense  # (B, H, T, D)

            dense_loss = F.mse_loss(dense_naction_pred, dense_naction, reduction="none")
            dense_loss = dense_loss.type(dense_loss.dtype)
            dense_loss = reduce(dense_loss, "b ... -> b (...)", "mean")
            dense_loss = dense_loss.mean()
            self.dense_loss = dense_loss

            loss += dense_loss
        return loss

    def forward(self, batch, flags):
        return self.compute_loss(batch, flags)

    def get_loss_components(self):
        return self.sparse_loss, self.smoothness_loss, self.dense_loss

    def has_dense(self):
        return self.flag_has_dense
