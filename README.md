# Deep Generative Models: Theory and Practice

#### Xinwei Zhang & An Zhang
#### June 30<sup>th</sup>, 2023

## Requirements
```
easydict      1.10
matplotlib    3.7.1
numpy         1.24.1
opencv-python 4.7.0.72
scipy         1.10.1
torch         2.0.1+cu118
torchvision   0.15.2+cu118
tqdm          4.65.0
```

## Structure
```
Train.ipynb       - Train VAE, GAN and WGAN models.
Diffusion.ipynb   - Train Diffusion models and get samples.
Generate.ipynb    - Generate samples from saved models (VAE, GAN and WGAN).
models.py         - Implementation of VAE, GAN, WGAN and loss functions.
unets.py          - UNet based network architecture for diffusion model.
diffusion.py      - Implementations of diffusion models and utilities for training diffusion models.
models            - Saved model state dictionaries (VAE, GAN and WGAN).
Diffusion_models  - Saved model state dictionaries (Diffusion), one every 20 training epochs.
samples           - Samples of VAE, GAN and WGAN.
diffusion_samples - Samples of Diffusion models, one every 20 training epochs.
history           - Loss histories of these models.
diffusion_samples - Samples of the diffusion model's denoising process.
MNIST             - The MNIST database.
```

## Training
The training processes and results are recorded in `Train.ipynb` and `Diffusion.ipynb`.

## Sampling
Call `get_sample` in `Generate.ipynb` with a loaded generative model (VAE, GAN or WGAN) to generate images. There are a number of examples in the notebook.

As for Diffusion Models, call `save_samples` in `Diffusion.ipynb` instead. This function produces a figure of `n_samples` denoising processes and saves it at the given directory.