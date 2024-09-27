torchrun --standalone --nproc_per_node=4 src/eval/eval.py \
    --outdir='eval_results/sdxl' \
    --seeds=0-29999 \
    --batch=32 \
    --repo_id='stabilityai/stable-diffusion-xl-base-1.0' \
    --resolution=1024 \
    --guidance_scale=7.5
    