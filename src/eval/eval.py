# Copyright (c) 2024, Mingyuan Zhou. All rights reserved.
#
# This work is licensed under APACHE LICENSE, VERSION 2.0
# You should have received a copy of the license along with this
# work. If not, see https://www.apache.org/licenses/LICENSE-2.0.txt


import os
import re
import click
import tqdm
import numpy as np
import torch
import PIL.Image

import torch
from diffusers import AutoPipelineForText2Image
from ..linfusion import LinFusion
from . import distributed as dist


#----------------------------------------------------------------------------
# Wrapper for torch.Generator that allows specifying a different random seed
# for each sample in a minibatch.

class StackedRandomGenerator:
    def __init__(self, device, seeds):
        super().__init__()
        self.generators = [torch.Generator(device).manual_seed(int(seed) % (1 << 32)) for seed in seeds]

    def randn(self, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randn(size[1:], generator=gen, **kwargs) for gen in self.generators])

    def randn_like(self, input):
        return self.randn(input.shape, dtype=input.dtype, layout=input.layout, device=input.device)

    def randint(self, *args, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randint(*args, size=size[1:], generator=gen, **kwargs) for gen in self.generators])

#----------------------------------------------------------------------------
# Parse a comma separated list of numbers or ranges and return a list of ints.
# Example: '1,2,5-10' returns [1, 2, 5, 6, 7, 8, 9, 10]

def parse_int_list(s):
    if isinstance(s, list): return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
        else:
            ranges.append(int(p))
    return ranges

def read_file_to_sentences(filename):
    # Initialize an empty list to store the sentences
    sentences = []

    # Open the file
    with open(filename, 'r', encoding='utf-8') as file:
        # Read each line from the file
        for line in file:
            # Strip newline and any trailing whitespace characters
            clean_line = line.strip()
            # Add the cleaned line to the list if it is not empty
            if clean_line:
                sentences.append(clean_line)
    
    return sentences

#----------------------------------------------------------------------------

def compress_to_npz(folder_path, num=50000):
    # Get the list of all files in the folder
    npz_path = f"{folder_path}.npz"
    file_names = os.listdir(folder_path)

    # Filter the list of files to include only images
    file_names = [file_name for file_name in file_names if file_name.endswith(('.png', '.jpg', '.jpeg'))]
    num = min(num, len(file_names))
    file_names = file_names[:num]

    # Initialize a dictionary to hold image arrays and their filenames
    samples = []

    # Iterate through the files, load each image, and add it to the dictionary with a progress bar
    for file_name in tqdm.tqdm(file_names, desc=f"Compressing images to {npz_path}"):
        # Create the full path to the image file
        file_path = os.path.join(folder_path, file_name)
        
        # Read the image using PIL and convert it to a NumPy array
        image = PIL.Image.open(file_path)
        image_array = np.asarray(image).astype(np.uint8)
        
        samples.append(image_array)
    samples = np.stack(samples)

    # Save the images as a .npz file
    np.savez(npz_path, arr_0=samples)
    print(f"Images from folder {folder_path} have been saved as {npz_path}")

#----------------------------------------------------------------------------


@click.command()
@click.option('--outdir',                  help='Where to save the output images', metavar='DIR',                   type=str, required=True)
@click.option('--seeds',                   help='Random seeds (e.g. 1,2,5-10)', metavar='LIST',                     type=parse_int_list, default='0-63', show_default=True)
@click.option('--subdirs',                 help='Create subdirectory for every 1000 seeds',                         is_flag=True)
@click.option('--batch', 'max_batch_size', help='Maximum batch size', metavar='INT',                                type=click.IntRange(min=1), default=16, show_default=True)
@click.option('--num', 'num_fid_samples',  help='Maximum num of images', metavar='INT',                             type=click.IntRange(min=1), default=30000, show_default=True)
@click.option('--text_prompts', 'text_prompts',   help='captions filename; the default [prompts/captions.txt] consists of 30k COCO2014 prompts', metavar='PATH|URL',         type=str, default='assets/captions.txt', show_default=True)
@click.option('--repo_id', 'repo_id',   help='diffusion pipeline filename', metavar='PATH|URL',                     type=str, default='runwayml/stable-diffusion-v1-5', show_default=True)
@click.option('--use_fp16',             help='Enable mixed-precision training', metavar='BOOL',                     type=bool, default=True, show_default=True)
@click.option('--use_bf16',             help='Enable mixed-precision training', metavar='BOOL',                     type=bool, default=False, show_default=True)
@click.option('--enable_compress_npz',         help='Enable compressinve npz', metavar='BOOL',                             type=bool, default=False, show_default=True)
@click.option('--num_steps_eval', 'num_steps_eval',      help='Set as 25 by default', metavar='INT',      type=click.IntRange(min=0), default=25, show_default=True)
@click.option('--guidance_scale', 'guidance_scale',      help='Scale of classifier-free guidance. Set as 7.5 by default', metavar='FLOAT',      type=click.FloatRange(min=1.0), default=7.5, show_default=True)
@click.option('--resolution', 'resolution',      help='Set as None by default, which means default resolution of the diffusion model', metavar='INT',      type=int, default=None, show_default=True)
@click.option('--custom_seed',             help='Enable custom seed', metavar='BOOL',                     type=bool, default=False, show_default=True)


def main(outdir, subdirs, seeds, max_batch_size, num_fid_samples, text_prompts,repo_id,device=torch.device('cuda'),use_fp16=True,use_bf16=False,enable_compress_npz=False,num_steps_eval=25,guidance_scale=7.5,resolution=None,custom_seed=False):
    
    dist.init()
    
    dtype=torch.float16 if use_fp16 else torch.float32
    dtype=torch.bfloat16 if use_bf16 else torch.float32
        
    captions = read_file_to_sentences(text_prompts)

    num_batches = ((len(seeds) - 1) // (max_batch_size * dist.get_world_size()) + 1) * dist.get_world_size()
    if not custom_seed:
        all_batches = torch.as_tensor(seeds).tensor_split(num_batches)
    else:
        seeds_idx = parse_int_list(f'0-{len(seeds)-1}')
        all_batches = torch.as_tensor(seeds_idx).tensor_split(num_batches)
    rank_batches = all_batches[dist.get_rank() :: dist.get_world_size()]
    
    # Rank 0 goes first.
    if dist.get_rank() != 0:
        torch.distributed.barrier()

    # Evaluate
    if use_fp16:
        pipeline = AutoPipelineForText2Image.from_pretrained(repo_id, variant='fp16').to(device, dtype)
    else:
        pipeline = AutoPipelineForText2Image.from_pretrained(repo_id).to(device, dtype)
    _ = LinFusion.construct_for(pipeline)
    resolution = resolution or pipeline.default_sample_size * pipeline.vae_scale_factor
    
    # Other ranks follow.
    if dist.get_rank() == 0:
        torch.distributed.barrier()
        
    # Loop over batches.
    dist.print0(f'Generating {len(seeds)} images to "{outdir}"...')
    for batch_seeds in tqdm.tqdm(rank_batches, unit='batch', disable=(dist.get_rank() != 0)):
        torch.distributed.barrier()
        batch_size = len(batch_seeds)
        if batch_size == 0:
            continue
        # Pick latents and labels.
        
        if not custom_seed:
            rnd = StackedRandomGenerator(device, batch_seeds)
        else:
            cseed= [seeds[i] for i in batch_seeds]
            rnd = StackedRandomGenerator(device, cseed)
        
        img_channels=4
        latents = rnd.randn([batch_size, img_channels, resolution // pipeline.vae_scale_factor, resolution // pipeline.vae_scale_factor], device=device, dtype=dtype)

        c = [captions[i] for i in batch_seeds]  # Index captions using list comprehension

        with torch.no_grad():
            images = pipeline(
                prompt=c,
                num_inference_steps=num_steps_eval,
                guidance_scale=guidance_scale,
                latents=latents
            ).images
            
        # Save images.
        for seed, image in zip(batch_seeds, images):
            image_dir = os.path.join(outdir, f'{seed-seed%1000:06d}') if subdirs else outdir
            os.makedirs(image_dir, exist_ok=True)
            image_path = os.path.join(image_dir, f'{seed:06d}.png')
            image.save(image_path)
    # Done.
    
    if enable_compress_npz:
        torch.distributed.barrier()
        if dist.get_rank() == 0:
            compress_to_npz(outdir, num_fid_samples)
        torch.distributed.barrier()
    dist.print0('Done.')
    
#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
