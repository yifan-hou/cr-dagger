import sys
import os
from typing import Dict, Callable, Tuple, List

SCRIPT_PATH = "/home/yifanhou/git/PyriteML/scripts"
sys.path.append(os.path.join(SCRIPT_PATH, '../'))


import numpy as np
import torch
import time
import dill
import hydra
import pickle
import time
import numpy as np
import numpy.typing as npt
from torch.utils.data import DataLoader
import robotmq as rmq

from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.dataset.base_dataset import BaseImageDataset, BaseDataset
from diffusion_policy.workspace.train_diffusion_unet_image_workspace import TrainDiffusionUnetImageWorkspace

# data_path = "/home/yifanhou/training_outputs/"
# ckpt_path = data_path + "2025.03.05_21.53.36_stow_no_force_202_stow_80/checkpoints/latest.ckpt"

data_path = "/shared_local/training_outputs/"
ckpt_path = data_path + "2025.03.07_13.55.19_stow_residual_residual_mlp/checkpoints/latest.ckpt"

# load checkpoint
if not ckpt_path.endswith('.ckpt'):
    ckpt_path = os.path.join(ckpt_path, 'checkpoints', 'latest.ckpt')
residual_payload = torch.load(open(ckpt_path, 'rb'), map_location='cpu', pickle_module=dill)

residual_cfg = residual_payload["cfg"]
residual_cls = hydra.utils.get_class(residual_cfg._target_)
residual_workspace = residual_cls(residual_cfg)
residual_workspace: BaseWorkspace
residual_workspace.load_payload(residual_payload, exclude_keys=None, include_keys=None)
residual_policy = residual_workspace.model
residual_obs_encoder = residual_workspace.obs_encoder
# residual_policy.eval().to(device)
# residual_obs_encoder.eval().to(device)
# residual_shape_meta = residual_cfg.task.shape_meta
# residual_normalizer = pickle.load(open(os.path.join(checkpoint_folder_path + pipeline_para["residual_ckpt_path"], "sparse_normalizer.pkl"), "rb"))


server = rmq.RMQServer(
    server_name="test_rmq_server", server_endpoint="ipc:///tmp/feeds/0"
)

print("Server created")

server.add_topic("test_checkpoints", 10)

input("Topic established. Press Enter to send data...")

# Serialize the checkpoint
start_time = time.time()

pickle_data = pickle.dumps(residual_policy)
# pickle_data = pickle.dumps(payload)
dump_end_time = time.time()
server.put_data("test_checkpoints", pickle_data)
send_end_time = time.time()
time.sleep(0.01)

print(
    f"[Server] Data size: {len(pickle_data) / 1024**2:.3f}MB. dump: {dump_end_time - start_time:.4f}s, send: {send_end_time - dump_end_time: .4f}s)"
)