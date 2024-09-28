torchrun --standalone --nproc_per_node=4 -m src.eval.eval \
    --outdir='eval_results/sdxl' \
    --seeds=0-29999 \
    --batch=8 \
    --repo_id='stabilityai/stable-diffusion-xl-base-1.0' \
    --resolution=1024 \
    --guidance_scale=7.5
    