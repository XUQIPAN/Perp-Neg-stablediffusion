
### [Project Page](https://perp-neg.github.io/) | [Paper](https://arxiv.org/abs/2304.04968) | [Huggingface][huggingface-demo]
[![][huggingface]][huggingface-demo]

<hr>


[colab]: <https://colab.research.google.com/assets/colab-badge.svg>
[huggingface]: <https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue>
[composable-demo]: <https://colab.research.google.com/github/energy-based-model/Compositional-Visual-Generation-with-Composable-Diffusion-Models-PyTorch/blob/main/notebooks/demo.ipynb>
[huggingface-demo]: <https://huggingface.co/spaces/Shuang59/Composable-Diffusion>

# PerpNeg-StableDiffusion

This is the repository for using Perp-Neg sampling with Stable Diffusion model, as presented in [Re-imagine the Negative Prompt Algorithm: Transform 2D Diffusion into 3D, alleviate Janus problem and Beyond.](https://Perp-Neg.github.io).



## Running code


**Run compositional ebm with diffusion model**
```
python scripts/image_sample_stable_diffusion.py --prompt="a boy wearing sunglasses|sunglasses with black frame" --weights="1|-1.5" --pipeline='cebm' --num_images=1
```

**Run PerpNeg**
```
python scripts/image_sample_stable_diffusion.py --prompt="a boy wearing sunglasses|sunglasses with black frame" --weights="1|-1.5" --pipeline='perpneg' --num_images=1
```

**Run PerpNeg-rotation**
```
python scripts/image_sample_stable_diffusion.py --prompt="a photo of lion, front view | a photo of lion, side view" --weights="1|-1.5|-1.5" --pipeline='rotation_perpneg' --num_images=1
```


If you find it useful for your work, please cite:
```
@article{armandpour2023re,
  title={Re-imagine the Negative Prompt Algorithm: Transform 2D Diffusion into 3D, alleviate Janus problem and Beyond},
  author={Armandpour, Mohammadreza and Sadeghian, Ali and Zheng, Huangjie and Sadeghian, Amir and Zhou, Mingyuan},
  journal={arXiv preprint arXiv:2304.04968},
  year={2023}
}
```
