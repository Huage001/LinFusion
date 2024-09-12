accelerate launch --num_processes 8 --multi_gpu --mixed_precision "bf16" --main_process_port 29500 \
  distill.py \
  --pretrained_model_name_or_path="stabilityai/stable-diffusion-2-1" \
  --mixed_precision="bf16" \
  --resolution=768 \
  --train_batch_size=6 \
  --gradient_accumulation_steps=2 \
  --dataloader_num_workers=6 \
  --learning_rate=1e-04 \
  --weight_decay=0. \
  --output_dir="ckpt/linfusion_sd2p1" \
  --save_steps=10000