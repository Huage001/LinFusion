import os
import copy
import numpy as np
import io
from datasets import load_dataset
from torchvision import transforms
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
from diffusers import StableDiffusionXLPipeline, DDPMScheduler
import functools

from ..linfusion import LinFusion


def get_submodule(model, module_name):
    return functools.reduce(getattr, module_name.split("."), model)


# Dataset
def get_laion_dataset(
    tokenizer,
    tokenizer_2,
    resolution=1024,
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
        example["input_ids_2"] = tokenizer_2(
            captions,
            padding="max_length",
            truncation=True,
            max_length=tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids
        return example

    dataset = dataset.map(get_blip_caption, with_indices=True, batched=True)

    process = transforms.Compose(
        [
            transforms.Resize(
                resolution, interpolation=transforms.InterpolationMode.BILINEAR
            ),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    def transform(example):
        batch = {}
        images = [
            Image.open(io.BytesIO(item["bytes"])).convert("RGB")
            for item in example["image"]
        ]
        batch["original_size"] = torch.stack(
            [torch.tensor([image.size[1], image.size[0]]) for image in images], dim=0
        )
        batch["target_size"] = torch.stack(
            [torch.tensor([resolution, resolution])] * len(images), dim=0
        )
        images = [process(image) for image in images]
        batch["crop_coords_top_left"] = torch.stack(
            [
                torch.tensor(
                    [
                        (image.shape[1] - resolution) // 2,
                        (image.shape[2] - resolution) // 2,
                    ]
                )
                for image in images
            ],
            dim=0,
        )
        batch["image"] = torch.stack(
            [
                image[
                    :,
                    coord[0] : coord[0] + resolution,
                    coord[1] : coord[1] + resolution,
                ]
                for image, coord in zip(images, batch["crop_coords_top_left"])
            ],
            dim=0,
        )
        batch["text_input_ids"] = torch.from_numpy(
            np.array(example["input_ids"])
        ).long()
        batch["text_input_ids_2"] = torch.from_numpy(
            np.array(example["input_ids_2"])
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
        default="distill_sdxl",
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
        default=1024,
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
        default=1,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=8,
        help="Gradient accumulation steps. Total bs=train_batch_size * gradient_accumulation_steps * num_gpus",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=1,
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
    parser.add_argument(
        "--mid_dim_scale",
        type=int,
        default=None,
        help="The scale of the mid_dim of the linear attention. `mid_dim = dim_n // mid_dim_scale`",
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
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        args.pretrained_model_name_or_path
    )
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )
    tokenizer = pipeline.tokenizer
    text_encoder = pipeline.text_encoder
    tokenizer_2 = pipeline.tokenizer_2
    text_encoder_2 = pipeline.text_encoder_2
    vae = pipeline.vae
    unet = pipeline.unet
    unet_teacher = copy.deepcopy(unet)
    # freeze parameters of models to save more memory
    unet.requires_grad_(False)
    unet_teacher.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    text_encoder_2.requires_grad_(False)

    all_attn_outputs = []
    all_attn_outputs_teacher = []

    if args.pretrained_linfusion_path is not None:
        # to construct a LinFusion model
        linfusion_model = LinFusion.construct_for(
            unet=unet,
            load_pretrained=True,
            pretrained_model_name_or_path=args.pretrained_linfusion_path,
        )
    else:
        linfusion_config = LinFusion.get_default_config(unet=unet)
        if args.mid_dim_scale is not None:
            for each in linfusion_config["modules_list"]:
                each["projection_mid_dim"] = each["dim_n"] // args.mid_dim_scale
        linfusion_model = LinFusion(**linfusion_config)
        linfusion_model.mount_to(unet=unet)

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
    text_encoder_2.to(accelerator.device, dtype=weight_dtype)
    linfusion_model.requires_grad_(True)

    # optimizer
    optimizer = torch.optim.AdamW(
        linfusion_model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    # dataloader
    train_dataset = get_laion_dataset(
        tokenizer=tokenizer, tokenizer_2=tokenizer_2, resolution=args.resolution
    )
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
                        batch["image"].to(accelerator.device, dtype=torch.float32)
                    ).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor
                    latents = latents.to(accelerator.device, dtype=weight_dtype)

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
                    
                    encoder_output = text_encoder(
                        batch["text_input_ids"].to(accelerator.device),
                        output_hidden_states=True,
                    )
                    text_embeds = encoder_output.hidden_states[-2]
                    encoder_output_2 = text_encoder_2(
                        batch["text_input_ids_2"].to(accelerator.device),
                        output_hidden_states=True,
                    )
                    pooled_text_embeds = encoder_output_2[0]
                    text_embeds_2 = encoder_output_2.hidden_states[-2]
                    text_embeds = torch.concat([text_embeds, text_embeds_2], dim=-1)

                    add_time_ids = torch.cat(
                        [
                            batch["original_size"],
                            batch["crop_coords_top_left"],
                            batch["target_size"],
                        ],
                        dim=1,
                    ).to(accelerator.device, dtype=weight_dtype)
                    unet_added_cond_kwargs = {
                        "text_embeds": pooled_text_embeds,
                        "time_ids": add_time_ids,
                    }

                    noise_pred_teacher = unet_teacher(
                        noisy_latents,
                        timesteps,
                        encoder_hidden_states=text_embeds,
                        added_cond_kwargs=unet_added_cond_kwargs,
                    ).sample

                noise_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=text_embeds,
                    added_cond_kwargs=unet_added_cond_kwargs,
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
                    noise_pred.float(), noise.float(), reduction="mean"
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
                    os.path.join(args.output_dir, f"linfusion-{global_step}"),
                    push_to_hub=False,
                )

            begin = time.perf_counter()


if __name__ == "__main__":
    main()
