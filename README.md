<div align="center">

# LinFusion
<a href="https://arxiv.org/abs/2409.02097"><img src="https://img.shields.io/badge/arXiv-2409.02097-A42C25.svg" alt="arXiv"></a> 
<a  href="https://lv-linfusion.github.io"><img src="https://img.shields.io/badge/ProjectPage-LinFusion-376ED2#376ED2.svg" alt="Home Page"></a>
<a href="https://huggingface.co/spaces/Huage001/LinFusion-SD-v1.5"><img src="https://img.shields.io/static/v1?label=HuggingFace&message=gradio demo&color=yellow"></a>
</div>


> **LinFusion: 1 GPU, 1 Minute, 16K Image**
> <br>
> [Songhua Liu](http://121.37.94.87/), 
> [Weuhao Yu](https://whyu.me/), 
> Zhenxiong Tan, 
> and 
> [Xinchao Wang](https://sites.google.com/site/sitexinchaowang/)
> <br>
> [Learning and Vision Lab](http://lv-nus.org/), National University of Singapore
> <br>

![](./assets/picture.png)

## 🔥News

**[2024/09/08]** We release codes for **16K** image generation [here](https://github.com/Huage001/LinFusion/blob/main/examples/inference/ultra_text2image_w_sdedit.ipynb)!

**[2024/09/05]** [Gradio demo](https://huggingface.co/spaces/Huage001/LinFusion-SD-v1.5) for SD-v1.5 is released! Text-to-image, image-to-image, and IP-Adapter are supported currently.

## Supported Models

1. `Yuanshi/LinFusion-1-5`: For Stable Diffusion 1.5 and its variants. <a href="https://huggingface.co/Yuanshi/LinFusion-1-5"><img src="https://img.shields.io/badge/%F0%9F%A4%97-LinFusion for 1.5-yellow"></a>


## Quick Start
* If you have not, install [PyTorch](https://pytorch.org/get-started/locally/) and [diffusers](https://huggingface.co/docs/diffusers/index).

* Clone this repo to your project directory:

  ``` bash
  git clone https://github.com/Huage001/LinFusion.git
  ```

* **You only need two lines!**

  ```diff
  from diffusers import AutoPipelineForText2Image
  import torch
  
  + from src.linfusion import LinFusion
  
  sd_repo = "Lykon/dreamshaper-8"
  
  pipeline = AutoPipelineForText2Image.from_pretrained(
      sd_repo, torch_dtype=torch.float16, variant="fp16"
  ).to(torch.device("cuda"))
  
  + linfusion = LinFusion.construct_for(pipeline)
  
  image = pipeline(
      "An astronaut floating in space. Beautiful view of the stars and the universe in the background.",
      generator=torch.manual_seed(123)
  ).images[0]
  ```
  `LinFusion.construct_for(pipeline)` will return a LinFusion model that matches the pipeline's structure. And this LinFusion model will **automatically mount to** the pipeline's forward function.

* `examples/inference/basic_usage.ipynb` shows a basic text-to-image example.

## Ultrahigh-Resolution Generation

* From the perspective of efficiency, our method supports high-resolution generation such as 16K images. Nevertheless, directly applying diffusion models trained on low resolutions for higher-resolution generation can result in content distortion and duplication. To tackle this challenge, we apply techniques in [SDEdit](https://huggingface.co/docs/diffusers/v0.30.2/en/api/pipelines/stable_diffusion/img2img#image-to-image). **The basic idea is to generate a low-resolution result at first, based on which we gradually upscale the image. Please refer to `examples/inference/ultra_text2image_w_sdedit.ipynb` for an example.** Note that 16K generation is only currently available for 80G GPUs. We will try to relax this constraint by implementing tiling strategies.
* We are working on integrating LinFusion with more advanced approaches that are dedicated on high-resolution extension!

## Training

```bash
bash ./examples/training/distill.sh
```


## ToDo
- [x] Stable Diffusion 1.5 support.
- [ ] Stable Diffusion 2.1 support. 
- [ ] Stable Diffusion XL support.
- [ ] Release training code for LinFusion.
- [ ] Release evaluation code for LinFusion.

## Citation

If you finds this repo is helpful, please consider cite:

```bib
@article{liu2024linfusion,
  title     = {LinFusion: 1 GPU, 1 Minute, 16K Image},
  author    = {Liu, Songhua and Yu, Weihao and Tan, Zhenxiong and Wang, Xinchao},
  year      = {2024},
  eprint    = {2409.02097},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
}
```
