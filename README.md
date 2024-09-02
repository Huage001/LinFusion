<div align="center">

# LinFusion
<a href="https://arxiv.org/"><img src="https://img.shields.io/badge/arXiv-xxxx.xxxx-A42C25.svg" alt="arXiv"></a>
<!-- TODO: Change the arxiv img -->
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

## Supported Models
1. `Yuanshi/LinFusion-1-5`: For Stable Diffusion 1.5 and its variants. <a href="https://huggingface.co/Yuanshi/LinFusion-1-5"><img src="https://img.shields.io/badge/%F0%9F%A4%97-LinFusion for 1.5-yellow"></a>


## Quick Start
* **Basic Usage of LinFusion:** (can be found in the `examples/basic_usage.ipynb` file.)


```python
from diffusers import AutoPipelineForText2Image
import torch

from src.linfusion import LinFusion
from src.tools import seed_everything

sd_repo = "Lykon/dreamshaper-8"

pipeline = AutoPipelineForText2Image.from_pretrained(
    sd_repo, torch_dtype=torch.float16, variant="fp16"
).to(torch.device("cuda"))

linfusion = LinFusion.construct_for(pipeline)

seed_everything(123)
image = pipeline(
	"An astronaut floating in space. Beautiful view of the stars and the universe in the background."
).images[0]
```
`LinFusion.construct_for(pipeline)` will return a LinFusion model that matches the pipeline's structure. And this LinFusion model will **automatically mount to** the pipeline's forward function.

## Customization
* **Specific the LinFusion model path:** 
```python
linfusion = LinFusion.construct_for(
    pipeline,
    pretrained_model_name_or_path = "[path to the model]"
)
```

* **Construct LinFusion model with specific parameters:** 

**Step 1.** Get the default parameters for some specific model:
```python
config = LinFusion.get_default_config(pipeline)
```
**Step 2.** Modify the parameters you want to change.
```python
config["modules_list"][0]["projection_mid_dim"] = 128
```

**Step 3.** Construct the LinFusion model by passing the modified parameters:
```python
linfusion = LinFusion(**config)
```

**Step 4.** Mount the LinFusion model to the pipeline:
```python
linfusion.mount_to(pipeline)
```



## ToDo
- [x] Stable Diffusion 1.5 support.
- [ ] Stable Diffusion 2.1 support. 
- [ ] Stable Diffusion XL support.
- [ ] Release training code for LinFusion.
- [ ] Release evaluation code for LinFusion.