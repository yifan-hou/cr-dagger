if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import hydra
import torch
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler
import copy
import random
import wandb
import pickle
import tqdm
import numpy as np

from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.model.common.lr_scheduler import get_scheduler
from diffusion_policy.model.vision.timm_obs_encoder_with_force import (
    TimmObsEncoderWithForce,
)
from diffusion_policy.model.residual.mlp import MLPResidual
from accelerate import Accelerator

OmegaConf.register_new_resolver("eval", eval, replace=True)

class TrainOnlineResidualMLPWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch']

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)
        
        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        self.obs_encoder: TimmObsEncoderWithForce = hydra.utils.instantiate(cfg.obs_encoder)
        if os.path.exists(cfg.base_policy_ckpt):
            print(f"==> loading pretrained weights from {cfg.base_policy_ckpt}")
            base_policy_ckpt = torch.load(cfg.base_policy_ckpt, weights_only=False)['state_dicts']['model']
            for key in self.obs_encoder.rgb_keys:
                key_prefix = f"obs_encoder.key_model_map.{key}"
                for model_key in base_policy_ckpt.keys():
                    if model_key.startswith(key_prefix):
                        self.obs_encoder.key_model_map[key].load_state_dict(
                            {model_key[len(key_prefix)+1:]: base_policy_ckpt[model_key]}
                        , strict=False)
                self.obs_encoder.key_model_map[key].requires_grad_(False)        # freeze the encoder
        obs_feature_dim = np.prod(self.obs_encoder.output_shape())
        # self.model: MLPResidual = hydra.utils.instantiate(cfg.model)
        print("==> obs_feature_dim: ", obs_feature_dim)
        self.model = MLPResidual(
            input_dim=obs_feature_dim,
            action_dim=cfg.model.action_dim,
            action_horizon=cfg.model.action_horizon,
            hidden_dims=cfg.model.hidden_dims,
            dropout=cfg.model.dropout,
        )
        self.sparse_normalizer = LinearNormalizer()

        train_obs_encoder_params = list()
        for param in self.obs_encoder.parameters():
            if param.requires_grad:
                train_obs_encoder_params.append(param)
        self.trainable_obs_encoders = {}
        for key in self.obs_encoder.wrench_keys:
            self.trainable_obs_encoders[key] = self.obs_encoder.key_model_map[key]

        print(f"obs_encoder params: {sum(p.numel() for p in train_obs_encoder_params)}")
        param_groups = [
            {"params": self.model.parameters()},
            {"params": train_obs_encoder_params}, 
        ]
        # self.optimizer = hydra.utils.instantiate(
        #     cfg.optimizer, params=param_groups)
        optimizer_cfg = OmegaConf.to_container(cfg.optimizer, resolve=True)
        optimizer_cfg.pop("_target_")
        self.optimizer = torch.optim.AdamW(params=param_groups, **optimizer_cfg)

        # # configure training state
        # self.optimizer = hydra.utils.instantiate(
        #     cfg.optimizer, params=self.model.parameters())

        # configure training state
        # These number will be affected if a checkpoint is loaded.
        self.global_step = 0
        self.epoch = 0

        self.topk_manager = None


    # Call this when there is sufficient data to train and compute normalizer
    # This is called once per life time of the workspace.
    # If loading checkpoint, this is called after loading the checkpoint.
    def train_preparation(self, dataset: BaseImageDataset):
        cfg = copy.deepcopy(self.cfg)
        wandb_cfg = OmegaConf.to_container(cfg.logging, resolve=True)
        wandb.init(
            project=wandb_cfg['project'],
            resume=wandb_cfg['resume'],
            mode=wandb_cfg['mode'],
            name=wandb_cfg['name'],
            tags=wandb_cfg['tags'],
            id=wandb_cfg['id'],
            group=wandb_cfg['group'],
        )

        # compute normalizer on the main process and save to disk
        sparse_normalizer_path = os.path.join(self.output_dir, "sparse_normalizer.pkl")
        self.sparse_normalizer, _ = dataset.get_normalizer()
        pickle.dump(self.sparse_normalizer, open(sparse_normalizer_path, "wb"))

        # configure checkpoint
        self.topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, "checkpoints"), **self.cfg.checkpoint.topk
        )

        # device transfer
        device = torch.device(self.cfg.training.device)
        print("==> training on device: ", device)
        self.obs_encoder.to(device)
        self.model.to(device)
        optimizer_to(self.optimizer, device)
        
    def run(self, dataset: BaseImageDataset):
        assert isinstance(dataset, BaseImageDataset)
        if dataset.sampling_weights is not None:
            num_samples = min(self.cfg.training.num_samples, len(dataset))
            print("[workspace] training with ", num_samples, " samples")
            weighted_random_sampler = WeightedRandomSampler(dataset.sampling_weights, num_samples, replacement=True)
            dataloader_cfg = self.cfg.dataloader.copy()
            dataloader_cfg.shuffle = False
            train_dataloader = DataLoader(dataset, **dataloader_cfg, sampler=weighted_random_sampler)
        else:
            train_dataloader = DataLoader(dataset, **self.cfg.dataloader)

        # configure validation dataset
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **self.cfg.val_dataloader)
        print(
            "train dataset:", len(dataset), "train dataloader:", len(train_dataloader)
        )
        print("val dataset:", len(val_dataset), "val dataloader:", len(val_dataloader))

        # prepare for training
        num_epochs = self.cfg.training.num_epochs
        lr_warmup_steps = self.cfg.training.lr_warmup_steps
        if self.global_step == 0:
            num_epochs = self.cfg.training.first_time_num_epochs
            lr_warmup_steps = self.cfg.training.first_time_lr_warmup_steps
            print("=====================================================")
            print(f"First time training. Running num_epochs: {num_epochs}")
            print("=====================================================")
        
        # configure lr scheduler, steps it every batch
        lr_scheduler = get_scheduler(
            self.cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=lr_warmup_steps,
            num_training_steps=(
                len(train_dataloader) * num_epochs) \
                    // self.cfg.training.gradient_accumulate_every,
            # last_epoch=self.global_step-1
        )

        # print batch size
        batch_size = self.cfg.dataloader.batch_size
        print(f"batch_size: {batch_size}")
        sample_batch = next(iter(train_dataloader))
        for key, attr in sample_batch["obs"]["sparse"].items():
            print("obs.sparse.key: ", key, attr.shape)
        print("obs.sparse: ", sample_batch["action"]["sparse"].shape)

        # print action dimension
        print("action: ", sample_batch["action"]["sparse"].shape)
        print("dataset.action_type: ", dataset.action_type)

        # save training batch for computing metrics for logging
        train_sampling_batch = None

        device = torch.device(self.cfg.training.device)

        # training loop
        log_path = os.path.join(self.output_dir, "logs.json.txt")
        with JsonLogger(log_path) as json_logger:
            for local_epoch_idx in range(num_epochs):
                self.model.train()

                step_log = dict()
                # ========= train for this epoch ==========
                train_losses = list()
                with tqdm.tqdm(
                    train_dataloader,
                    desc=f"Training epoch {self.epoch}",
                    leave=False,
                    mininterval=self.cfg.training.tqdm_interval_sec,
                ) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):
                        # device transfer
                        batch = dict_apply(
                            batch, lambda x: x.to(device, non_blocking=True)
                        )

                        # always use the latest batch
                        # except for the last batch of an epoch (last batch might not have full batch size)
                        if (
                            batch_idx == 0
                            or batch["action"]["sparse"].shape[0] == batch_size
                        ):
                            train_sampling_batch = batch

                        obs_dict_sparse = batch["obs"]["sparse"]
                        nobs_sparse = self.sparse_normalizer.normalize(obs_dict_sparse)
                        nobs_sparse = dict_apply(nobs_sparse, lambda x: x.to(device))
                        sparse_nobs_encode = self.obs_encoder(nobs_sparse)

                        action_sparse = self.sparse_normalizer["action"].normalize(batch["action"]["sparse"]).to(device)

                        # compute loss
                        raw_loss = self.model.compute_loss(sparse_nobs_encode, action_sparse)  # TODO: compute loss

                        loss = raw_loss / self.cfg.training.gradient_accumulate_every
                        loss.backward()
                        # accelerator.backward(raw_loss)

                        # compute gradient for logging
                        if self.cfg.training.log_gradient_norm:
                            sparse_grads = [
                                param.grad.detach().flatten()
                                for param in self.model.model_sparse.parameters()
                                if param.grad is not None
                            ]
                            sparse_norm = (
                                torch.cat(sparse_grads).norm()
                                if len(sparse_grads) > 0
                                else 0
                            )

                        # step optimizer
                        if (
                            self.global_step % self.cfg.training.gradient_accumulate_every
                            == 0
                        ):
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                            lr_scheduler.step()

                        # logging
                        raw_loss_cpu = raw_loss.item()
                        tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                        train_losses.append(raw_loss_cpu)
                        step_log = {
                            "loss": raw_loss_cpu,
                            "global_step": self.global_step,
                            "epoch": self.epoch,
                            "lr": lr_scheduler.get_last_lr()[0],
                        }
                        if self.cfg.training.log_gradient_norm:
                            step_log["sparse_gradient_norm"] = sparse_norm

                        is_last_batch = batch_idx == (len(train_dataloader) - 1)
                        if not is_last_batch:
                            # log of last step is combined with validation and rollout
                            wandb.log(step_log, step=self.global_step)
                            json_logger.log(step_log)
                            self.global_step += 1

                        if (self.cfg.training.max_train_steps is not None) and batch_idx >= (
                            self.cfg.training.max_train_steps - 1
                        ):
                            break

                # at the end of each epoch
                # replace train_loss with epoch average
                train_loss = np.mean(train_losses)
                step_log["train_loss"] = train_loss

                # ========= eval for this epoch ==========
                self.model.eval()

                def log_action_mse(step_log, category, pred_action, gt_action):
                    step_log[f"{category}_action_mse_error"] = (
                        torch.nn.functional.mse_loss(
                            pred_action, gt_action
                        )
                    )
                    # step_log[f'{category}_sparse_naction_mse_error_pos'] = torch.nn.functional.mse_loss(pred_naction_sparse[..., :3], gt_naction_sparse[..., :3])
                    # step_log[f'{category}_sparse_naction_mse_error_rot'] = torch.nn.functional.mse_loss(pred_naction_sparse[..., 3:9], gt_naction_sparse[..., 3:9])
                    if pred_action.shape[-1] == 3:
                        return

                    step_log[f"{category}_action_mse_error"] = (
                        torch.nn.functional.mse_loss(
                            pred_action[..., :9], gt_action[..., :9]
                        )
                    )
                    if pred_action.shape[-1] == 15:
                        step_log[f"{category}_force_mse_error"] = (
                            torch.nn.functional.mse_loss(
                                pred_action[..., 9:15], gt_action[..., 9:15]
                            )
                        )
                    # step_log[f"{category}_vt_action_mse_error"] = (
                    #     torch.nn.functional.mse_loss(
                    #         pred_action[..., 9:18], gt_action[..., 9:18]
                    #     )
                    # )
                    # step_log[f"{category}_stiffness_mse_error"] = (
                    #     torch.nn.functional.mse_loss(
                    #         pred_action[..., 18], gt_action[..., 18]
                    #     )
                    # )

                # Compute action prediction MSE error
                if (self.epoch % self.cfg.training.sample_every) == 0:
                    with torch.no_grad():
                        # sample trajectory from training set, and evaluate difference
                        batch = dict_apply(
                            train_sampling_batch,
                            lambda x: x.to(device, non_blocking=True),
                        )
                        gt_action = self.sparse_normalizer["action"].normalize(batch["action"]["sparse"]).to(device)
                        obs_dict_sparse = batch["obs"]["sparse"]
                        nobs_sparse = self.sparse_normalizer.normalize(obs_dict_sparse)
                        nobs_sparse = dict_apply(nobs_sparse, lambda x: x.to(device))
                        sparse_nobs_encode = self.obs_encoder(nobs_sparse)
                        pred_action = self.model(sparse_nobs_encode)

                        log_action_mse(step_log, "train", pred_action, gt_action)

                        if len(val_dataloader) > 0:
                            val_sampling_batch = next(iter(val_dataloader))
                            batch = dict_apply(
                                val_sampling_batch,
                                lambda x: x.to(device, non_blocking=True),
                            )
                            gt_action = self.sparse_normalizer["action"].normalize(batch["action"]["sparse"]).to(device)
                            obs_dict_sparse = batch["obs"]["sparse"]
                            nobs_sparse = self.sparse_normalizer.normalize(obs_dict_sparse)
                            nobs_sparse = dict_apply(nobs_sparse, lambda x: x.to(device))
                            sparse_nobs_encode = self.obs_encoder(nobs_sparse)
                            pred_action = self.model(sparse_nobs_encode)
                            log_action_mse(step_log, "val", pred_action, gt_action)

                        del batch
                        del gt_action
                        del pred_action

                # checkpoint
                print("==> epoch: ", self.epoch, ", train_loss: ", train_loss)
                print("==> self.cfg.training.checkpoint_every: ", self.cfg.training.checkpoint_every)
                print("==> self.epoch % self.cfg.training.checkpoint_every: ", self.epoch % self.cfg.training.checkpoint_every)
                if (self.epoch % self.cfg.training.checkpoint_every) == 0:
                    # checkpointing
                    if self.cfg.checkpoint.save_last_ckpt:
                        print("==> saving lastest checkpoint for epoch: ", self.epoch)
                        self.save_checkpoint() # latest checkpoint
                    if self.cfg.checkpoint.save_last_snapshot:
                        self.save_snapshot()

                    # sanitize metric names
                    metric_dict = dict()
                    for key, value in step_log.items():
                        new_key = key.replace("/", "_")
                        metric_dict[new_key] = value

                    # We can't copy the last checkpoint here
                    # since save_checkpoint uses threads.
                    # therefore at this point the file might have been empty!
                    topk_ckpt_path = self.topk_manager.get_ckpt_path(metric_dict)

                    if topk_ckpt_path is not None:
                        print("==> saving topk checkpoint: ", topk_ckpt_path)
                        self.save_checkpoint(path=topk_ckpt_path) # epoch=xxxx-train_loss=xxxx.ckpt

                # ========= eval end for this epoch ==========
                # end of epoch
                # log of last step is combined with validation and rollout
                wandb.log(step_log, step=self.global_step)
                json_logger.log(step_log)
                self.global_step += 1
                self.epoch += 1

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")), 
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = TrainOnlineResidualMLPWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
