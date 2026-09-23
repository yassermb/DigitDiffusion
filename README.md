# Diffusion Model for MNIST

## Overview

This project implements a **Denoising Diffusion Probabilistic Model (DDPM)** from scratch in PyTorch, trained on the MNIST handwritten digit dataset (28×28 grayscale). The model learns to generate realistic handwritten digits by reversing a gradual noise-addition (forward) process.

---

## Repository structure

```
Diffusion/
├── config.py          # All hyperparameters and paths
├── diffusion.py       # Noise schedules & forward process utilities
├── model.py           # U-Net architecture
├── dataset.py         # MNIST data loading (train/val/test splits)
├── train.py           # Training loop, EMA, sampling, validation, testing
├── SUMMARY.md         # This file
├── checkpoints/       # Saved model weights (best.pt, final.pt)
├── samples/           # Generated digit grids per epoch
├── results/           # Final generated samples
├── runs/              # TensorBoard log directory
└── data/              # MNIST dataset (auto-downloaded)
```

---

## Architecture

### U-Net noise prediction network

The model is a U-Net that predicts the noise $\varepsilon_\theta(x_t, t)$ added during the forward process.

| Component | Details |
|---|---|
| **Input** | Noisy image $x_t$ (1×28×28) + time-step $t$ |
| **Time embedding** | Learned MLP: normalise $t/T \to [0,1]$ -> 4-layer MLP (dim=256) |
| **Encoder** | 3 resolution levels: 128 -> 256 -> 512 channels |
| **Residual blocks** | 2 per level; GroupNorm -> SiLU -> Conv3×3 -> +time -> GroupNorm -> SiLU -> Dropout -> Conv3×3 |
| **Down-sampling** | Strided 3×3 convolution (stride=2) |
| **Bottleneck** | 2 residual blocks at 512 channels (7×7 spatial) |
| **Decoder** | Mirror of encoder with skip connections |
| **Up-sampling** | Nearest-neighbour interpolation + 3×3 convolution |
| **Output** | GroupNorm -> SiLU -> 1×1 Conv -> 1 channel |
| **Total parameters** | **54,969,857** (~55.0M) |

### U-Net spatial dimensions

```
28×28 -> [ResBlock×2 @ 128] -> Downsample -> 14×14
14×14 -> [ResBlock×2 @ 256] -> Downsample ->  7×7
 7×7  -> [ResBlock×2 @ 512] (bottleneck)
 7×7  -> Upsample -> 14×14 -> [ResBlock×3 @ 256]  (+ skip)
14×14 -> Upsample -> 28×28 -> [ResBlock×3 @ 128]  (+ skip)
28×28 -> Output head -> 1 channel
```

---

## Diffusion process

### Forward process (adding noise)

Given a clean data (*e.g.* image) $x_0$, we define a <font color="green">**Markov chain**</font> that adds Gaussian noise over $T$ steps:

$$q(x_t \mid x_{t-1}) = \mathcal{N}\!\left(x_t;\; \sqrt{1 - \beta_t}\, x_{t-1},\; \beta_t\, \mathbf{I}\right)$$

where $\beta_1, \beta_2, \ldots, \beta_T$ is a <font color="green">**variance schedule**</font> (small positive values that increase over time).

As we have seen in the course, we can jump to any step without iterating by defining a new variable, called $\bar{\alpha}_t$. The goal is to sample $x_t$ directly from $x_0$ without iterating:

$$\alpha_t = 1 - \beta_t, \qquad \bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s, \qquad \beta_t = 1 - \frac{\bar{\alpha}_t}{\bar{\alpha}_{t-1}}$$

$$x_t = \sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1 - \bar{\alpha}_t}\, \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, \mathbf{I})$$


### Reverse Process (denoising / generation)

We want to learn the <font color="#3E74D1">**approximate reverse process**</font>: given $x_t$, recover $x_{t-1}$.

$$p_\theta(x_{t-1} \mid x_t) = \mathcal{N}\!\left(x_{t-1};\; \mu_\theta(x_t, t),\; \sigma_t^2\, \mathbf{I}\right)$$

This is <font color="#3E74D1">**approximate posterior**</font> which wants to be as similar as possible to the <font color="#FFBD16">**true posterior (true reverse process)**</font>:

$$
q(x_{t-1}|x_t,x_0) = \mathcal{N}(x_{t-1}; \tilde\mu_t(x_t, x_0), \tilde\beta_t I)
$$

$$
\tilde\mu_t(x_t, x_0) = \frac{\sqrt{\bar\alpha_{t-1}}\beta_t}{1-\bar\alpha_t}x_0 + \frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}x_t, \qquad \tilde\beta_t = \frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\beta_t
$$

As we saw in the course, the optimal mean is:

$$\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}}\, \varepsilon_\theta(x_t, t) \right)$$

where $\varepsilon_\theta$ is a neural network that <font color="green">predicts the noise that was added.</font>

### The noise schedule

The noise schedule or variance schedule $\{\beta_t\}_{t=1}^T$ controls how quickly we add noise. We have two options here:

1) Linear variance schedule
$$\beta_t = \beta_{\text{start}} + \frac{t-1}{T-1}\left(\beta_{\text{end}} - \beta_{\text{start}}\right), \qquad t = 1,\dots,T$$

2) Cosine schedule from [Nichol & Dhariwal (2021)](https://arxiv.org/abs/2102.09672). It distributes noise more uniformly across timesteps for better quality:

$$\bar{\alpha}_t = \frac{f(t)}{f(0)}, \quad f(t) = \cos\!\left(\frac{t/T + s}{1+s} \cdot \frac{\pi}{2}\right)^2$$

with offset $s = 0.008$. This produces better sample quality than the original linear schedule.

### Training objective

This is the DDPM loss we saw in the course which is <font color="green">beautifully simple: just MSE between true and predicted noise!</font>

$$
L(\theta) = \mathbb{E}_q\left[\frac{\beta_t^2}{2\sigma_t^2\alpha_t(1-\bar\alpha_t)}\left\|\epsilon - \epsilon_\theta\left(x_t, t\right)\right\|^2\right]
$$

[Ho, Jain & Abbeel (2020)](https://arxiv.org/abs/2006.11239) has further simplified this loss to improved sample quality and simpler to implement for training.

$$L_\mathrm{simple}(\theta) = \mathbb{E}_{t,x_0,\epsilon}\left[\left\|\epsilon - \epsilon_\theta\left(\sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon, t\right)\right\|^2\right]$$

$$L_{\text{simple}}(\theta) = \mathbb{E}_{t,\, x_0,\, \varepsilon}\!\left[\left\| \varepsilon - \varepsilon_\theta(x_t, t) \right\|^2\right]$$

<font color="green">**This means sample a random timestep, noise the image, predict the noise, minimise the error.**</font>

---

## Training configuration

### Parameters

| Parameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | $2 \times 10^{-4}$ with warmup (500 steps) + cosine decay |
| Batch size | 128 |
| Epochs | 50 |
| Noise schedule | Cosine (Nichol & Dhariwal, 2021) |
| Gradient clipping | Max norm = 1.0 |
| EMA decay | 0.9999 (updated **per batch**, not per epoch) |
| Dropout | 0.1 |
| Seed | 42 |

### Data splits

| Split | Samples | Batches |
|---|---|---|
| Train | 54,000 | 421 |
| Validation | 6,000 | 47 |
| Test | 10,000 | 79 |


---

## Key Implementation Details

1. **Only U-Net architecture is used**: The model achieves good quality on 28×28 MNIST using only convolutional residual blocks.

2. **Exponential Moving Average (EMA)**: An EMA copy of the model weights (decay=0.9999) is updated every training batch (not per-epoch) and used for sampling and evaluation, producing smoother and higher-quality samples.

3. **Cosine noise schedule**: Following Nichol & Dhariwal (2021), a cosine schedule distributes noise better across time-steps, improving generation quality compared to the original linear schedule.

4. **LR warmup + cosine decay**: The learning rate warms up linearly over 500 steps, then decays via cosine annealing - stabilising early training and enabling fine-grained learning in later epochs.

5. **Learned time embeddings**: Time-step $t$ is normalised to $[0,1]$ and mapped through a simple 4-layer MLP before being added to each residual block.

6. **GroupNorm + SiLU**: Every convolutional layer uses GroupNorm (32 groups) and SiLU activation, following modern best practices for generative models.

7. **TensorBoard logging**: Training/validation losses, learning rate, sample grids, forward process visualisation, and the real image reference batch are all logged for interactive monitoring.

---

## How to Reproduce

```bash
# Create a conda environment called diffusion and install necessary components, including torch.

# Activate environment
conda activate diffusion

# Train the model (takes around 1 to two hours on GPU)
python train.py

# Monitor in TensorBoard
tensorboard --logdir runs --port 6008
```

---

## Files Description
**config.py**: Centralised configuration — all hyperparameters, paths, and constants
**diffusion.py**: `DiffusionSchedule` class: computes β, α, ᾱ schedules; `q_sample` for forward noising; `q_posterior_mean_variance` for reverse process
**model.py**: `UNet` with `LearnedTimeEmbedding`, `ResidualBlock`, `Downsample`, `Upsample`
**dataset.py**: `get_dataloaders()` — MNIST with [-1,1] normalisation and 90/10 train/val split
**train.py**: `main()` orchestrator; `train_one_epoch()`, `validate()`, `test()`, `sample()` functions; `EMA` class

---

## References

- Ho, J., Jain, A., & Abbeel, P. (2020). *Denoising Diffusion Probabilistic Models*. NeurIPS 2020.
- Nichol, A. & Dhariwal, P. (2021). *Improved Denoising Diffusion Probabilistic Models*. ICML 2021.
- Ronneberger, O., Fischer, P., & Brox, T. (2015). *U-Net: Convolutional Networks for Biomedical Image Segmentation*. MICCAI 2015.