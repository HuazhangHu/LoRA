# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Fine-tuning script for Stable Diffusion for text2image with support for LoRA."""

import argparse
import logging
import math
import os
import random
import shutil
from pathlib import Path
from PIL import Image
# from tensorboardX import SummaryWriter

import datasets
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from huggingface_hub import upload_folder
from packaging import version
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, DiffusionPipeline, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import compute_snr
from diffusers.utils import check_min_version, convert_state_dict_to_diffusers, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
from training_cfg import load_training_config

from dataloader_train import TrainData
# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.26.0.dev0")

logger = get_logger(__name__, log_level="INFO")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('cfg', type=str)
    args = parser.parse_args()

    return args

def get_accelerator(cfg):
    logging_dir = Path(cfg.log_dir)

    accelerator_project_config = ProjectConfiguration(
        project_dir=cfg.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        mixed_precision=cfg.mixed_precision,
        log_with="tensorboard",
        project_config=accelerator_project_config, # 输出config log
    )

    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if cfg.seed is not None:
        set_seed(cfg.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if cfg.output_dir is not None:
            os.makedirs(cfg.output_dir, exist_ok=True)

    return accelerator

def get_models(cfg):
    # Load scheduler, tokenizer and models.
    ### loading scheduler
    noise_scheduler = DDPMScheduler.from_pretrained(
        cfg.pretrained_model_name_or_path, subfolder="scheduler")
    ### model loading
    tokenizer = CLIPTokenizer.from_pretrained(
        cfg.pretrained_model_name_or_path, subfolder="tokenizer"
    )
    text_encoder = CLIPTextModel.from_pretrained(
        cfg.pretrained_model_name_or_path, subfolder="text_encoder"
    )
    vae = AutoencoderKL.from_pretrained(
        cfg.pretrained_model_name_or_path, subfolder="vae",
    )
    unet = UNet2DConditionModel.from_pretrained(
        cfg.pretrained_model_name_or_path, subfolder="unet"
    )

    return noise_scheduler, tokenizer, text_encoder, vae, unet

def lora_adapter(cfg):
    lora_config = LoraConfig(
        r=cfg.rank,
        lora_alpha=cfg.rank,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )

    return lora_config


def main():
    args = get_args()
    cfg_path = args.cfg

    cfg = load_training_config(cfg_path) #  TrainingConfig Class

    accelerator = get_accelerator(cfg)
    
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    # Load scheduler, tokenizer and models.
    noise_scheduler, tokenizer, text_encoder, vae, unet = get_models(cfg)
    # freeze parameters of models to save more memory
    unet.requires_grad_(False)  # Freeze
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    # Freeze the unet parameters before adding adapters for training
    for param in unet.parameters():
        param.requires_grad_(False)

    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32 #  full precision
    if accelerator.mixed_precision == "fp16": # half-precision
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move unet, vae and text_encoder to device and cast to weight_dtype
    unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

     #lora_adapter
    lora_config = lora_adapter(cfg)
    # Add adapter and make sure the trainable params are in float32.
    unet.add_adapter(lora_config)
    
    if cfg.mixed_precision == "fp16":
        for param in unet.parameters():
            # only upcast trainable parameters (LoRA) into fp32
            if param.requires_grad:
                param.data = param.to(torch.float32)

    if cfg.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers
            xformers_version = version.parse(xformers.__version__)
            # print("xformers_version:",xformers_version)
            unet.enable_xformers_memory_efficient_attention() #函数会尝试启用 xformers 来提升 Attention 的效率
        else:
            raise ValueError(
                "xformers is not available. Make sure it is installed correctly")

    lora_layers = filter(lambda p: p.requires_grad, unet.parameters())

    # only optimize lora layers
    optimizer_cls = torch.optim.AdamW
    optimizer = optimizer_cls(
        lora_layers,
        lr=cfg.learning_rate,
        betas=(cfg.adam_beta1, cfg.adam_beta2),
        weight_decay=cfg.adam_weight_decay,
        eps=cfg.adam_epsilon,
    )

    # Get the datasets: you can either provide your own training and evaluation files (see below)
    # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).

    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    with accelerator.main_process_first():
        # Set the training transforms 确保数据集的预处理和转换只执行一次，并且由主进程负责。
        # train_dataset = dataset["train"].with_transform(preprocess_train)
        train_dataset = TrainData(cfg)

    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=cfg.train_batch_size,
        num_workers=cfg.dataloader_num_workers,
    )

    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(  
        len(train_dataloader) / cfg.gradient_accumulation_steps) ##每个epoch训练多少步
    max_train_steps = cfg.n_epochs * num_update_steps_per_epoch ## 总共的训练步数

    lr_scheduler = get_scheduler(
        cfg.lr_scheduler_name,
        optimizer=optimizer
    )

    # Prepare everything with our `accelerator`.
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / cfg.gradient_accumulation_steps)
    # Afterwards we recalculate our number of training epochs
    num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers(cfg.log_dir, config=vars(args))
        # tensorboard_tracker = accelerator.get_tracker("tensorboard")
        # writer = SummaryWriter(log_dir=os.path.join('tensor_log/', cfg.log_dir))

    # Train!
    total_batch_size = cfg.train_batch_size * \
        accelerator.num_processes * cfg.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {num_train_epochs}")
    logger.info(
        f"  Instantaneous batch size per device = {cfg.train_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(
        f"  Gradient Accumulation steps = {cfg.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if cfg.resume_from_checkpoint:
        if cfg.resume_from_checkpoint != "latest":
            path = os.path.basename(cfg.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(cfg.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{cfg.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(cfg.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )
    # from torch.utils.tensorboard import SummaryWriter
    # tb_writer = SummaryWriter(log_dir='./logs')

    for epoch in range(first_epoch, num_train_epochs):
        unet.train()
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                # print(batch[0].shape) # torch.Size([1, 3, 512, 512])
                # Convert images to latent space
                latents = vae.encode(batch[0].to(
                    dtype=weight_dtype)).latent_dist.sample()
                
                latents = latents * vae.config.scaling_factor  # vae.config.scaling_factor =0.18215

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                if cfg.noise_offset:
                    # https://www.crosslabs.org//blog/diffusion-with-offset-noise
                    noise += cfg.noise_offset * torch.randn(
                        (latents.shape[0], latents.shape[1], 1, 1), device=latents.device
                    )

                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(
                    latents, noise, timesteps)

                # Get the text embedding for conditioning  ## 文本条件注入
                encoder_hidden_states = text_encoder(batch[1])[0].to(torch.float32)
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(
                        latents, noise, timesteps)
                else:
                    raise ValueError(
                        f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                # Predict the noise residual and compute loss
                model_pred = unet(noisy_latents, timesteps,
                                  encoder_hidden_states).sample

                loss = F.mse_loss(model_pred.float(),
                                  target.float(), reduction="mean")

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(
                    loss.repeat(cfg.train_batch_size)).mean()
                train_loss += avg_loss.item() / cfg.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)  ## 梯度回传
                if accelerator.sync_gradients:  #  可以保证所有 GPU 都同步了梯度再执行后续代码
                    params_to_clip = lora_layers
                    accelerator.clip_grad_norm_( #需要裁剪梯度的参数，防止梯度爆炸
                        params_to_clip, cfg.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                # tensorboard_tracker.log({"loss": loss.item()}, step=global_step)
                # writer.add_scalars('train/loss',{"train_loss": train_loss}, global_step)
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0
                if accelerator.is_main_process:
                    # tb_writer.add_scalar('loss', loss.item(), global_step)
                    if global_step % cfg.checkpointing_steps == 0:
                        save_path = os.path.join(
                            cfg.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)

                        unwrapped_unet = accelerator.unwrap_model(unet)
                        unet_lora_state_dict = convert_state_dict_to_diffusers(
                            get_peft_model_state_dict(unwrapped_unet)
                        )

                        StableDiffusionPipeline.save_lora_weights(
                            save_directory=save_path,
                            unet_lora_layers=unet_lora_state_dict,
                            safe_serialization=True,
                        )

                        logger.info(f"Saved state to {save_path}")

            logs = {"step_loss": loss.detach().item(
            ), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            # writer.add_scalars('learning_rate', {"learning_rate": optimizer.state_dict()['param_groups'][0]['lr']}, epoch)

            if global_step >= max_train_steps:
                writer.close()
                break

    # Save the lora layers
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = unet.to(torch.float32)

        unwrapped_unet = accelerator.unwrap_model(unet)
        unet_lora_state_dict = convert_state_dict_to_diffusers(
            get_peft_model_state_dict(unwrapped_unet))
        StableDiffusionPipeline.save_lora_weights(
            save_directory=cfg.output_dir,
            unet_lora_layers=unet_lora_state_dict,
            safe_serialization=True,
            weight_name=cfg.ckpt_name + '.safetensor'
        )

    accelerator.end_training()


if __name__ == "__main__":
    main()