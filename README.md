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


## Basic Usage
* **Basic Usage of LinFusion:** can be found in the `src/linfusion.py` file. 
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


## ToDo
- [x] Stable Diffusion 1.5 support.
- [ ] Stable Diffusion 2.1 support. 
- [ ] Stable Diffusion XL support.
- [ ] Release training code for LinFusion.
- [ ] Release evaluation code for LinFusion.