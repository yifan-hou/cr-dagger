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
from diffusion_policy.model.common.lr_scheduler import get_scheduler
from diffusion_policy.model.vision.timm_obs_encoder_force_residual import (
    TimmObsEncoderForceResidual,
)
from diffusion_policy.model.residual.transformer import Transformer
from accelerate import Accelerator

OmegaConf.register_new_resolver("eval", eval, replace=True)

class TrainResidualTransformerWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch']

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)
        
        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        self.obs_encoder: TimmObsEncoderForceResidual = hydra.utils.instantiate(cfg.obs_encoder)
        if os.path.exists(cfg.base_policy_ckpt):
            print(f"==> loading pretrained weights from {cfg.base_policy_ckpt}")
            base_policy_ckpt = torch.load(cfg.base_policy_ckpt)['state_dicts']['model']
            for key in self.obs_encoder.rgb_keys:
                key_prefix = f"obs_encoder.key_model_map.{key}"
                for model_key in base_policy_ckpt.keys():
                    if model_key.startswith(key_prefix):
                        self.obs_encoder.key_model_map[key].load_state_dict(
                            {model_key[len(key_prefix)+1:]: base_policy_ckpt[model_key]}
                        , strict=False)
        obs_feature_dim = np.prod(self.obs_encoder.output_shape())
        # self.model: MLPResidual = hydra.utils.instantiate(cfg.model)
        print("==> obs_feature_dim: ", obs_feature_dim)
        self.model = Transformer(
            d=cfg.model.d,
            h=cfg.model.h,
            d_ff=cfg.model.d_ff,
            a = cfg.model.a,
            num_layers=cfg.model.num_layers,
            dropout=cfg.model.dropout,
            horizon=cfg.model.horizon
        )

        self.flag_has_dense = "dense" in cfg.task.shape_meta.sample.action
        print("==> model has_dense: ", self.flag_has_dense)
        # configure training state
        # obs_encorder_lr = cfg.optimizer.lr
        # if cfg.obs_encoder["reduce_pretrained_lr"]:
        #     obs_encorder_lr *= 0.1
        #     print("==> reduce pretrained obs_encorder's lr")
        # pretraiend_obs_encorder_params = list()
        # for key in self.obs_encoder.rgb_keys:
        #     for param in self.obs_encoder.key_model_map[key].parameters():
        #         if param.requires_grad:
        #             pretraiend_obs_encorder_params.append(param)
        # print(f"obs_encorder params: {len(pretraiend_obs_encorder_params)}")
        param_groups = [
            {"params": self.model.parameters()},
            {"params": self.obs_encoder.parameters()},  # TODO
            # {"params": pretraiend_obs_encorder_params, "lr": obs_encorder_lr},
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
        self.global_step = 0
        self.epoch = 0

    def run(self):
        cfg = copy.deepcopy(self.cfg)

        accelerator = Accelerator(log_with="wandb")
        wandb_cfg = OmegaConf.to_container(cfg.logging, resolve=True)
        wandb_cfg.pop("project")
        accelerator.init_trackers(
            project_name=cfg.logging.project,
            config=OmegaConf.to_container(cfg, resolve=True),
            init_kwargs={"wandb": wandb_cfg},
        )

        # resume training
        if cfg.training.resume:
            lastest_ckpt_path = self.get_checkpoint_path()
            if lastest_ckpt_path.is_file():
                print(f"Resuming from checkpoint {lastest_ckpt_path}")
                self.load_checkpoint(path=lastest_ckpt_path)

        # configure dataset
        dataset: BaseImageDataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(dataset, BaseImageDataset)
        train_dataloader = DataLoader(dataset, **cfg.dataloader)

        # compute normalizer on the main process and save to disk
        sparse_normalizer_path = os.path.join(self.output_dir, "sparse_normalizer.pkl")
        dense_normalizer_path = os.path.join(self.output_dir, "dense_normalizer.pkl")
        if accelerator.is_main_process:
            sparse_normalizer, dense_normalizer = dataset.get_normalizer()
            pickle.dump(sparse_normalizer, open(sparse_normalizer_path, "wb"))
            if dense_normalizer is not None:
                pickle.dump(dense_normalizer, open(dense_normalizer_path, "wb"))

        # load normalizer on all processes
        accelerator.wait_for_everyone()
        sparse_normalizer = pickle.load(open(sparse_normalizer_path, "rb"))
        if self.flag_has_dense:
            dense_normalizer = pickle.load(open(dense_normalizer_path, "rb"))
        else:
            dense_normalizer = None

        # configure validation dataset
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)
        print(
            "train dataset:", len(dataset), "train dataloader:", len(train_dataloader)
        )
        print("val dataset:", len(val_dataset), "val dataloader:", len(val_dataloader))

        # configure lr scheduler
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(
                len(train_dataloader) * cfg.training.num_epochs) \
                    // cfg.training.gradient_accumulate_every,
            # pytorch assumes stepping LRScheduler every epoch
            # however huggingface diffusers steps it every batch
            last_epoch=self.global_step-1
        )

        # configure checkpoint
        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, "checkpoints"), **cfg.checkpoint.topk
        )

        # device transfer
        device = torch.device(cfg.training.device)
        print("==> training on device: ", device)
        self.obs_encoder.to(device)
        self.model.to(device)
        optimizer_to(self.optimizer, device)

        # accelerator
        train_dataloader, val_dataloader, self.model, self.optimizer, lr_scheduler = (
            accelerator.prepare(
                train_dataloader,
                val_dataloader,
                self.model,
                self.optimizer,
                lr_scheduler,
            )
        )

        # print batch size
        batch_size = cfg.dataloader.batch_size
        print(f"batch_size: {batch_size}")
        sample_batch = next(iter(train_dataloader))
        for key, attr in sample_batch["obs"]["sparse"].items():
            print("obs.sparse.key: ", key, attr.shape)
        print("obs.sparse: ", sample_batch["action"]["sparse"].shape)
        if self.flag_has_dense:
            for key, attr in sample_batch["obs"]["dense"].items():
                print("obs.dense.key: ", key, attr.shape)
            print("obs.dense: ", sample_batch["action"]["dense"].shape)

        # print action dimension
        print("action: ", sample_batch["action"]["sparse"].shape)
        print("dataset.action_type: ", dataset.action_type)
        action_dimension = sample_batch["action"]["sparse"].shape[-1]

        # save batch for sampling
        train_sampling_batch = None

        if cfg.training.debug:
            cfg.training.num_epochs = 2
            cfg.training.max_train_steps = 3
            cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 1
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1

        args = {
            "dense_traj_cond_use_gt": False,
        }       

        # training loop
        log_path = os.path.join(self.output_dir, "logs.json.txt")
        with JsonLogger(log_path) as json_logger:
            for local_epoch_idx in range(cfg.training.num_epochs):
                self.model.train()

                step_log = dict()
                # ========= train for this epoch ==========
                if cfg.training.freeze_encoder:
                    self.model.obs_encoder.eval()
                    self.model.obs_encoder.requires_grad_(False)

                if local_epoch_idx < cfg.training.gt_traj_num_epochs:
                    args["dense_traj_cond_use_gt"] = True
                else:
                    args["dense_traj_cond_use_gt"] = False

                if local_epoch_idx > cfg.training.dense_training_start_epochs:
                    args["start_training_dense"] = True
                else:
                    args["start_training_dense"] = False

                normalization_weight = 0
                progress = local_epoch_idx / cfg.training.num_epochs
                if progress > cfg.training.normalization_schedule[0]:
                    if progress < cfg.training.normalization_schedule[1]:
                        normalization_weight = (
                            progress - cfg.training.normalization_schedule[0]
                        ) / (
                            cfg.training.normalization_schedule[1]
                            - cfg.training.normalization_schedule[0]
                        )
                    else:
                        normalization_weight = 1
                args["normalization_weight"] = (
                    normalization_weight * cfg.training.normalization_weight
                )

                train_losses = list()
                with tqdm.tqdm(
                    train_dataloader,
                    desc=f"Training epoch {self.epoch}",
                    leave=False,
                    mininterval=cfg.training.tqdm_interval_sec,
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
                        obs_dict_sparse_without_time = dict()
                        for key in obs_dict_sparse:
                            if "time_stamps" not in key:
                                obs_dict_sparse_without_time[key] = obs_dict_sparse[key]
                        nobs_sparse = sparse_normalizer.normalize(obs_dict_sparse_without_time)
                        for key in obs_dict_sparse:
                            if "time_stamps" in key:
                                nobs_sparse[key] = obs_dict_sparse[key]
                        nobs_sparse = dict_apply(nobs_sparse, lambda x: x.to(device))
                        sparse_nobs_encode, time_encode = self.obs_encoder(nobs_sparse)

                        action_sparse = sparse_normalizer["action"].normalize(batch["action"]["sparse"]).to(device)

                        # compute loss
                        raw_loss = self.model.compute_loss(sparse_nobs_encode, time_encode, action_sparse) # action_sparse: [B, 5, 3]

                        # loss = raw_loss / cfg.training.gradient_accumulate_every
                        # loss.backward()
                        accelerator.backward(raw_loss)

                        # clip gradient
                        model_unwrapped = accelerator.unwrap_model(self.model)
                        if cfg.training.sparse_gradient_norm_limit > 0:
                            torch.nn.utils.clip_grad_norm_(
                                model_unwrapped.parameters(),
                                cfg.training.sparse_gradient_norm_limit,
                            )
                        # if (
                        #     self.flag_has_dense
                        #     and cfg.training.dense_gradient_norm_limit > 0
                        # ):
                        #     torch.nn.utils.clip_grad_norm_(
                        #         model_unwrapped.model_dense.parameters(),
                        #         cfg.training.dense_gradient_norm_limit,
                        #     )

                        # compute gradient for logging
                        if cfg.training.log_gradient_norm:
                            model_unwrapped = accelerator.unwrap_model(self.model)
                            sparse_grads = [
                                param.grad.detach().flatten()
                                for param in model_unwrapped.parameters()
                                if param.grad is not None
                            ]
                            sparse_norm = (
                                torch.cat(sparse_grads).norm()
                                if len(sparse_grads) > 0
                                else 0
                            )

                            if self.flag_has_dense:
                                dense_grads = [
                                    param.grad.detach().flatten()
                                    for param in model_unwrapped.model_dense.parameters()
                                    if param.grad is not None
                                ]
                                dense_norm = (
                                    torch.cat(dense_grads).norm()
                                    if len(dense_grads) > 0
                                    else 0
                                )

                        # step optimizer
                        if (
                            self.global_step % cfg.training.gradient_accumulate_every
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
                        if cfg.training.log_gradient_norm:
                            step_log["sparse_gradient_norm"] = sparse_norm
                            if self.flag_has_dense:
                                step_log["dense_gradient_norm"] = dense_norm

                        is_last_batch = batch_idx == (len(train_dataloader) - 1)
                        if not is_last_batch:
                            # log of last step is combined with validation and rollout
                            accelerator.log(step_log, step=self.global_step)
                            json_logger.log(step_log)
                            self.global_step += 1

                        if (cfg.training.max_train_steps is not None) and batch_idx >= (
                            cfg.training.max_train_steps - 1
                        ):
                            break

                # at the end of each epoch
                # replace train_loss with epoch average
                train_loss = np.mean(train_losses)
                step_log["train_loss"] = train_loss

                # ========= eval for this epoch ==========
                policy = accelerator.unwrap_model(self.model)
                policy.eval()

                # # run rollout
                # if (self.epoch % cfg.training.rollout_every) == 0:
                #     runner_log = env_runner.run(policy)
                #     # log all
                #     step_log.update(runner_log)

                # run validation
                # if (self.epoch % cfg.training.val_every) == 0 and len(val_dataloader) > 0 and accelerator.is_main_process:
                #     with torch.no_grad():
                #         val_losses = list()
                #         with tqdm.tqdm(val_dataloader, desc=f"Validation epoch {self.epoch}",
                #                 leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                #             for batch_idx, batch in enumerate(tepoch):
                #                 batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                #                 loss = self.model(batch)
                #                 val_losses.append(loss)
                #                 if (cfg.training.max_val_steps is not None) \
                #                     and batch_idx >= (cfg.training.max_val_steps-1):
                #                     break
                #         if len(val_losses) > 0:
                #             val_loss = torch.mean(torch.tensor(val_losses)).item()
                #             # log epoch average validation loss
                #             step_log['val_loss'] = val_loss

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
                    step_log[f"{category}_vt_action_mse_error"] = (
                        torch.nn.functional.mse_loss(
                            pred_action[..., 9:18], gt_action[..., 9:18]
                        )
                    )
                    step_log[f"{category}_stiffness_mse_error"] = (
                        torch.nn.functional.mse_loss(
                            pred_action[..., 18], gt_action[..., 18]
                        )
                    )

                # run diffusion sampling on a training batch
                if (
                    self.epoch % cfg.training.sample_every
                ) == 0 and accelerator.is_main_process:
                    with torch.no_grad():
                        # sample trajectory from training set, and evaluate difference
                        batch = dict_apply(
                            train_sampling_batch,
                            lambda x: x.to(device, non_blocking=True),
                        )
                        gt_action = sparse_normalizer["action"].normalize(batch["action"]["sparse"]).to(device)
                        obs_dict_sparse = batch["obs"]["sparse"]
                        obs_dict_sparse_without_time = dict()
                        for key in obs_dict_sparse:
                            if "time_stamps" not in key:
                                obs_dict_sparse_without_time[key] = obs_dict_sparse[key]
                        nobs_sparse = sparse_normalizer.normalize(obs_dict_sparse_without_time)
                        for key in obs_dict_sparse:
                            if "time_stamps" in key:
                                nobs_sparse[key] = obs_dict_sparse[key]
                        nobs_sparse = dict_apply(nobs_sparse, lambda x: x.to(device))
                        sparse_nobs_encode, time_encode = self.obs_encoder(nobs_sparse)
                        pred_action = policy.predict_actions(sparse_nobs_encode, time_encode, gt_action.shape)

                        log_action_mse(step_log, "train", pred_action, gt_action)

                        if len(val_dataloader) > 0:
                            val_sampling_batch = next(iter(val_dataloader))
                            batch = dict_apply(
                                val_sampling_batch,
                                lambda x: x.to(device, non_blocking=True),
                            )
                            gt_action = sparse_normalizer["action"].normalize(batch["action"]["sparse"]).to(device)
                            obs_dict_sparse = batch["obs"]["sparse"]
                            obs_dict_sparse_without_time = dict()
                            for key in obs_dict_sparse:
                                if "time_stamps" not in key:
                                    obs_dict_sparse_without_time[key] = obs_dict_sparse[key]
                            nobs_sparse = sparse_normalizer.normalize(obs_dict_sparse_without_time)
                            for key in obs_dict_sparse:
                                if "time_stamps" in key:
                                    nobs_sparse[key] = obs_dict_sparse[key]
                            nobs_sparse = dict_apply(nobs_sparse, lambda x: x.to(device))
                            sparse_nobs_encode, time_encode = self.obs_encoder(nobs_sparse)
                            pred_action = policy.predict_actions(sparse_nobs_encode, time_encode, gt_action.shape)

                        del batch
                        del gt_action
                        del pred_action

                # checkpoint
                if (
                    self.epoch % cfg.training.checkpoint_every
                ) == 0 and accelerator.is_main_process:
                    # unwrap the model to save ckpt
                    model_ddp = self.model
                    self.model = accelerator.unwrap_model(self.model)

                    # checkpointing
                    if cfg.checkpoint.save_last_ckpt:
                        self.save_checkpoint()
                    if cfg.checkpoint.save_last_snapshot:
                        self.save_snapshot()

                    # sanitize metric names
                    metric_dict = dict()
                    for key, value in step_log.items():
                        new_key = key.replace("/", "_")
                        metric_dict[new_key] = value

                    # We can't copy the last checkpoint here
                    # since save_checkpoint uses threads.
                    # therefore at this point the file might have been empty!
                    topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)

                    if topk_ckpt_path is not None:
                        self.save_checkpoint(path=topk_ckpt_path)

                    # recover the DDP model
                    self.model = model_ddp
                # ========= eval end for this epoch ==========
                # end of epoch
                # log of last step is combined with validation and rollout
                accelerator.log(step_log, step=self.global_step)
                json_logger.log(step_log)
                self.global_step += 1
                self.epoch += 1

        accelerator.end_training()

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")), 
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = TrainResidualTransformerWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
