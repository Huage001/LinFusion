import os
import numpy as np
import cv2
import io
from datasets import load_dataset
import argparse
from pathlib import Path
import json
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer
import functools

from src.linfusion import LinFusion


def get_submodule(model, module_name):
    return functools.reduce(getattr, module_name.split("."), model)


# Dataset
def get_laion_dataset(
    tokenizer,
    resolution=512,
    path="bhargavsdesai/laion_improved_aesthetics_6.5plus_with_images",
):
    dataset = load_dataset(path, split="train")
    with open(
        "./assets/laion_improved_aesthetics_6.5plus_with_images_blip_captions.json"
    ) as read_file:
        all_captions = json.load(read_file)

    def get_blip_caption(example, idx):
        captions = [all_captions[item] for item in idx]
        example["input_ids"] = tokenizer(
            captions,
            padding="max_length",
            truncation=True,
            max_length=tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids
        return example

    dataset = dataset.map(get_blip_caption, with_indices=True, batched=True)

    def process(image):
        img = np.array(image)
        img = cv2.resize(img, (resolution, resolution), interpolation=cv2.INTER_CUBIC)
        img = np.array(img).astype(np.float32)
        img = img / 127.5 - 1.0
        return torch.from_numpy(img).permute(2, 0, 1)

    def transform(example):
        batch = {}
        images = [
            Image.open(io.BytesIO(item["bytes"])).convert("RGB")
            for item in example["image"]
        ]
        batch["image"] = torch.stack([process(image) for image in images], dim=0)
        batch["text_input_ids"] = torch.from_numpy(
            np.array(example["input_ids"])
        ).long()
        return batch

    dataset.set_transform(transform)

    return dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_linfusion_path",
        type=str,
        default=None,
        help="Path to pretrained ip adapter model. If not specified weights are initialized randomly.",
    )
    parser.add_argument(
        "--data_json_file",
        type=str,
        default=None,
        help="Training data",
    )
    parser.add_argument(
        "--data_root_path",
        type=str,
        default=None,
        help="Training data root path",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="linear_attn_tune_attn",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=("The resolution for input images"),
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate to use.",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0, help="Weight decay to use."
    )
    parser.add_argument("--num_train_epochs", type=int, default=300)
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=6,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=2,
        help="Gradient accumulation steps. Total bs=train_batch_size * gradient_accumulation_steps * num_gpus",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=8,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=10000,
        help=("Save a checkpoint of the training state every X updates"),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank",
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


def main():
    args = parse_args()
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir
    )

    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer"
    )
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder"
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae"
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet"
    )
    unet_teacher = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet"
    )
    # freeze parameters of models to save more memory
    unet.requires_grad_(False)
    unet_teacher.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    all_attn_outputs = []
    all_attn_outputs_teacher = []

    # to construct a LinFusion model
    linfusion_model = LinFusion.construct_for(
        unet=unet,
        load_pretrained=args.pretrained_linfusion_path is not None,
        pretrained_model_name_or_path=args.pretrained_linfusion_path,
    )

    def student_forward_hook(module, input, output):
        all_attn_outputs.append(output)

    def teacher_forward_hook(module, input, output):
        all_attn_outputs_teacher.append(output)

    for sub_module in linfusion_model.modules_list:
        sub_module_name = sub_module["module_name"]
        student_module = get_submodule(unet, sub_module_name)
        teacher_module = get_submodule(unet_teacher, sub_module_name)
        student_module.register_forward_hook(student_forward_hook)
        teacher_module.register_forward_hook(teacher_forward_hook)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    unet_teacher.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    linfusion_model.requires_grad_(True)

    # optimizer
    optimizer = torch.optim.AdamW(
        linfusion_model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    # dataloader
    train_dataset = get_laion_dataset(tokenizer=tokenizer, resolution=args.resolution)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    # Prepare everything with our `accelerator`.
    linfusion_model, unet, optimizer, train_dataloader = accelerator.prepare(
        linfusion_model, unet, optimizer, train_dataloader
    )

    global_step = 0
    for epoch in range(0, args.num_train_epochs):
        begin = time.perf_counter()
        for step, batch in enumerate(train_dataloader):
            load_data_time = time.perf_counter() - begin
            with accelerator.accumulate(linfusion_model):
                # Convert images to latent space
                with torch.no_grad():
                    latents = vae.encode(
                        batch["image"].to(accelerator.device, dtype=weight_dtype)
                    ).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(
                    0,
                    noise_scheduler.num_train_timesteps,
                    (bsz,),
                    device=latents.device,
                )
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                with torch.no_grad():
                    encoder_hidden_states = text_encoder(
                        batch["text_input_ids"].to(accelerator.device)
                    )[0]
                    noise_pred_teacher = unet_teacher(
                        noisy_latents,
                        timesteps,
                        encoder_hidden_states=encoder_hidden_states,
                    ).sample

                noise_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                ).sample

                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(
                        f"Unknown prediction type {noise_scheduler.config.prediction_type}"
                    )

                loss_noise = F.mse_loss(
                    noise_pred.float(), target.float(), reduction="mean"
                )
                loss_kd = F.mse_loss(
                    noise_pred.float(), noise_pred_teacher.float(), reduction="mean"
                )
                loss_feat = sum(
                    [
                        F.mse_loss(feat.float(), feat_teacher.float())
                        for feat, feat_teacher in zip(
                            all_attn_outputs, all_attn_outputs_teacher
                        )
                    ]
                ) / len(all_attn_outputs_teacher)
                loss = loss_noise + loss_kd * 0.5 + loss_feat * 0.5

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = (
                    accelerator.gather(loss.repeat(args.train_batch_size)).mean().item()
                )
                avg_loss_kd = (
                    accelerator.gather(loss_kd.repeat(args.train_batch_size))
                    .mean()
                    .item()
                )
                avg_loss_feat = (
                    accelerator.gather(loss_feat.repeat(args.train_batch_size))
                    .mean()
                    .item()
                )
                avg_loss_noise = (
                    accelerator.gather(loss_noise.repeat(args.train_batch_size))
                    .mean()
                    .item()
                )

                # Backpropagate
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
                all_attn_outputs.clear()
                all_attn_outputs_teacher.clear()

                if accelerator.is_main_process:
                    print(
                        "Epoch {}, step {}, data_time: {}, time: {}, step_loss: {}, step_loss_noise: {}, step_loss_kd: {}, step_loss_feat: {}".format(
                            epoch,
                            step,
                            load_data_time,
                            time.perf_counter() - begin,
                            avg_loss,
                            avg_loss_noise,
                            avg_loss_kd,
                            avg_loss_feat,
                        )
                    )

            global_step += 1

            if global_step % args.save_steps == 0 and accelerator.is_main_process:
                # Save model checkpoint
                linfusion_model.save_pretrained(
                    os.path.join(args.output_dir, f"linfusion-{global_step}"), push_to_hub=False
                )

            begin = time.perf_counter()


if __name__ == "__main__":
    main()
