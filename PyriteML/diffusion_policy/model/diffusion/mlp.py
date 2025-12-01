from typing import Union
import logging
import torch
import torch.nn as nn
import einops
from einops.layers.torch import Rearrange
from collections import OrderedDict

from diffusion_policy.model.diffusion.conv1d_components import (
    Downsample1d,
    Upsample1d,
    Conv1dBlock,
)
from diffusion_policy.model.diffusion.positional_embedding import SinusoidalPosEmb

logger = logging.getLogger(__name__)


class MLP(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_channels,  # e.g [256, 128]
    ):
        super().__init__()

        # TO try:
        #   1. residual implementation with feedforward
        #   2. experiment with layers 2~8
        #   3. experiment with hidden layer dimension 512~2048
        self.blocks = nn.Sequential(
            OrderedDict(
                [
                    ("dense1", nn.Linear(in_channels, hidden_channels[0])),
                    ("act1", nn.ReLU()),
                    ("dense2", nn.Linear(hidden_channels[0], hidden_channels[1])),
                    ("act2", nn.ReLU()),
                    ("output", nn.Linear(hidden_channels[1], out_channels)),
                ]
            )
        )

    def forward(
        self,
        x,  # (B, :)
    ):
        """
        returns:
        out : [ batch_size x action_dimension ]
        """
        out = self.blocks(x)
        return out


class MLP_conditioned_with_time_encoding(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        action_horizon,  # used for time encoding
    ):
        super().__init__()

        # get encoder of time stamp
        dsed = in_channels
        time_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed, max_value=action_horizon),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )

        # TO try:
        #   1. residual implementation with feedforward
        #   2. experiment with layers 2~8
        #   3. experiment with hidden layer dimension 512~2048
        self.blocks = nn.Sequential(
            OrderedDict(
                [
                    ("dense1", nn.Linear(in_channels, 256)),
                    ("act1", nn.ReLU()),
                    ("dense2", nn.Linear(256, 256)),
                    ("act2", nn.ReLU()),
                    ("output", nn.Linear(256, out_channels)),
                ]
            )
        )

        self.time_encoder = time_encoder

    def forward(
        self,
        x,  # (B, :)
        cond,  # (B, :)
        timestep,  # (1)
    ):
        """
        returns:
        out : [ batch_size x action_dimension ]
        """
        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=x.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(x.device)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(x.shape[0])
        time_encoding = self.time_encoder(timesteps)  # (B, diffusion_step_embed_dim)

        global_feature = torch.cat([x, cond], axis=-1)
        global_feature = global_feature + time_encoding

        out = self.blocks(global_feature)
        return out
