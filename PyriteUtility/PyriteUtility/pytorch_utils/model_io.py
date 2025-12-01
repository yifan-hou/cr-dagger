import sys
import os

SCRIPT_PATH = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(SCRIPT_PATH, "../../.."))

import torch
import dill
import hydra

from PyriteML.diffusion_policy.workspace.base_workspace import BaseWorkspace
from PyriteML.diffusion_policy.workspace.train_diffusion_unet_image_finetune_workspace import (
    TrainDiffusionUnetImageFinetuneWorkspace,
)


def load_policy(ckpt_path, device):
    # load checkpoint
    if not ckpt_path.endswith(".ckpt"):
        ckpt_path = os.path.join(ckpt_path, "checkpoints", "latest.ckpt")
    payload = torch.load(open(ckpt_path, "rb"), map_location="cpu", pickle_module=dill)
    cfg = payload["cfg"]
    # print("model_name:", cfg.policy.obs_encoder.model_name)
    # print("dataset_path:", cfg.task.dataset.dataset_path)

    cls = hydra.utils.get_class(cfg._target_)
    # TODO(Yifan) Get policy independent from workspace
    workspace = cls(cfg)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=['optimizer'], include_keys=None)

    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model
    policy.num_inference_steps = (
        cfg.policy.num_inference_steps
    )  # DDIM inference iterations

    policy.eval().to(device)
    policy.reset()
    return policy, cfg.task.shape_meta


def serialize_model(ckpt_path):
    policy, shape_meta = load_policy(ckpt_path, "cuda")
    sm = torch.jit.script(policy)
    sm.save(ckpt_path.replace(".ckpt", ".pt"))


# testing
class MyModule(torch.nn.Module):
    def __init__(self, N, M):
        super(MyModule, self).__init__()
        self.weight = torch.nn.Parameter(torch.rand(N, M))

    def forward(self, input):
        if input.sum() > 0:
            output = self.weight.mv(input)
        else:
            output = self.weight + input
        return output

    def hahaha(self, x):
        x = x + 1
        return x


if __name__ == "__main__":
    policy, meta = load_policy(
        "/shared_local/training_outputs/test_kl_belt_new/checkpoints/latest.ckpt", "cuda"
    )