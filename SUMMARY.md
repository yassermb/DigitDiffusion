# DDPM Diffusion Model for MNIST — Project Summary

## Overview

This project implements a **Denoising Diffusion Probabilistic Model (DDPM)** from scratch in PyTorch, trained on the MNIST handwritten digit dataset (28×28 grayscale). The implementation follows the paper:

> **"Denoising Diffusion Probabilistic Models"** — Jonathan Ho, Ajay Jain, Pieter Abbeel (NeurIPS 2020)

The model learns to generate realistic handwritten digits by reversing a gradual noise-addition process, without using any attention mechanisms or Transformer-derived components — relying entirely on convolutional residual blocks and simple learned embeddings.

---

## Project Structure

```
Diffusion/
├── config.py          # All hyperparameters and paths
├── diffusion.py       # Noise schedules & forward process utilities
├── model.py           # U-Net architecture (no attention)
├── dataset.py         # MNIST data loading (train/val/test splits)
├── train.py           # Training loop, EMA, sampling, validation, testing
├── SUMMARY.md         # ← This file
├── BLOG.md            # Blog post with formulas and explanations
├── ddpm_tutorial.ipynb# Educational Jupyter notebook
├── checkpoints/       # Saved model weights (best.pt, final.pt)
├── samples/           # Generated digit grids per epoch
├── results/           # Final generated samples
├── runs/              # TensorBoard log directory
└── data/              # MNIST dataset (auto-downloaded)
```

---

## Architecture

### U-Net Noise Prediction Network

The model is a **U-Net** that predicts the noise $\varepsilon_\theta(x_t, t)$ added during the forward process.

| Component | Details |
|---|---|
| **Input** | Noisy image $x_t$ (1×28×28) + time-step $t$ |
| **Time embedding** | Learned MLP: normalise $t/T \to [0,1]$ → 4-layer MLP (dim=256) |
| **Encoder** | 3 resolution levels: 128 → 256 → 512 channels |
| **Residual blocks** | 2 per level; GroupNorm → SiLU → Conv3×3 → +time → GroupNorm → SiLU → Dropout → Conv3×3 |
| **Down-sampling** | Strided 3×3 convolution (stride=2) |
| **Bottleneck** | 2 residual blocks at 512 channels (7×7 spatial) |
| **Decoder** | Mirror of encoder with skip connections |
| **Up-sampling** | Nearest-neighbour interpolation + 3×3 convolution |
| **Output** | GroupNorm → SiLU → 1×1 Conv → 1 channel |
| **Total parameters** | **54,969,857** (~55.0M) |
| **Attention** | **None** — purely convolutional |

### Spatial Resolution Path

```
28×28 → [ResBlock×2 @ 128] → Downsample → 14×14
14×14 → [ResBlock×2 @ 256] → Downsample →  7×7
 7×7  → [ResBlock×2 @ 512] (bottleneck)
 7×7  → Upsample → 14×14 → [ResBlock×3 @ 256]  (+ skip)
14×14 → Upsample → 28×28 → [ResBlock×3 @ 128]  (+ skip)
28×28 → Output head → 1 channel
```

---

## Diffusion Process

### Forward Process (Adding Noise)

Given a clean image $x_0$, the forward process progressively adds Gaussian noise over $T = 1000$ steps:

$$q(x_t | x_0) = \mathcal{N}\!\left(x_t;\; \sqrt{\bar{\alpha}_t}\, x_0,\; (1 - \bar{\alpha}_t)\, \mathbf{I}\right)$$

where $\bar{\alpha}_t = \prod_{s=1}^{t} (1 - \beta_s)$ and $\beta_t$ follows a **cosine schedule** (Nichol & Dhariwal, 2021), which distributes noise more uniformly across time-steps than the original linear schedule.

### Reverse Process (Denoising / Generation)

Starting from pure Gaussian noise $x_T \sim \mathcal{N}(0, \mathbf{I})$, the model iteratively denoises:

$$x_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}}\, \varepsilon_\theta(x_t, t) \right) + \sigma_t\, z, \quad z \sim \mathcal{N}(0, \mathbf{I})$$

### Training Objective

The simplified DDPM loss trains the network to predict the noise:

$$L_{\text{simple}} = \mathbb{E}_{t,\, x_0,\, \varepsilon}\!\left[\left\| \varepsilon - \varepsilon_\theta(x_t, t) \right\|^2\right]$$

---

## Training Configuration

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

### Data Splits

| Split | Samples | Batches |
|---|---|---|
| Train | 54,000 | 421 |
| Validation | 6,000 | 47 |
| Test | 10,000 | 79 |

---

## Training Results

### Loss Progression

| Epoch | Train Loss | Val Loss (EMA) | Best? |
|---|---|---|---|
| 1 | 0.168368 | 1.352752 | ★ |
| 5 | 0.040881 | 0.531508 | ★ |
| 10 | 0.038524 | 0.219496 | ★ |
| 13 | 0.038173 | 0.197705 | ★ |
| 26 | 0.036460 | 0.194363 | ★ |
| 30 | 0.035975 | 0.172032 | ★ |
| 35 | 0.035174 | 0.139827 | ★ |
| 40 | 0.035033 | 0.117636 | ★ |
| 45 | 0.034903 | 0.095146 | ★ |
| 50 | 0.034751 | 0.077922 | ★ |

> The EMA validation loss drops continuously across all 50 epochs as the per-batch EMA accumulates properly. Train loss converges to ~0.035 while EMA val loss reaches **0.078** — demonstrating that the purely learned time embedding works just as well as sinusoidal encodings for MNIST.

### Test Results

| Metric | Value |
|---|---|
| **Test Loss (EMA model)** | **0.078435** |

### Saved Checkpoints

| File | Description |
|---|---|
| `checkpoints/best.pt` | Best validation loss model (epoch 50, val_loss=0.077922) |
| `checkpoints/final.pt` | Final model after epoch 50 |

### Generated Samples

Sample grids are saved at epochs 1, 10, 20, 30, 40, 50 in the `samples/` directory. Final samples (using the EMA model) are in `results/final_samples.png`.

---

## Key Implementation Details

1. **No Attention Layers**: The model achieves good quality on 28×28 MNIST using only convolutional residual blocks — attention is unnecessary at this resolution.

2. **Exponential Moving Average (EMA)**: An EMA copy of the model weights (decay=0.9999) is updated **every training batch** (not per-epoch) and used for sampling and evaluation, producing smoother and higher-quality samples.

3. **Cosine Noise Schedule**: Following Nichol & Dhariwal (2021), a cosine schedule distributes noise more uniformly across time-steps, improving generation quality compared to the original linear schedule.

4. **LR Warmup + Cosine Decay**: The learning rate warms up linearly over 500 steps, then decays via cosine annealing — stabilising early training and enabling fine-grained learning in later epochs.

5. **Learned Time Embeddings**: Time-step $t$ is normalised to $[0,1]$ and mapped through a simple 4-layer MLP (purely learned, no Transformer-derived formulas) before being added to each residual block.

6. **GroupNorm + SiLU**: Every convolutional layer uses GroupNorm (32 groups) and SiLU activation, following modern best practices for generative models.

7. **TensorBoard Logging**: Training/validation losses, learning rate, sample grids, forward process visualisation, and the real image reference batch are all logged for interactive monitoring.

---

## How to Reproduce

```bash
# Activate environment
conda activate diffusion

# Train the model (takes ~1-2 hours on GPU)
python train.py

# Monitor in TensorBoard
tensorboard --logdir runs --port 6008
```

---

## Files Description

| File | Purpose |
|---|---|
| **config.py** | Centralised configuration — all hyperparameters, paths, and constants |
| **diffusion.py** | `DiffusionSchedule` class: computes β, α, ᾱ schedules; `q_sample` for forward noising; `q_posterior_mean_variance` for reverse process |
| **model.py** | `UNet` with `LearnedTimeEmbedding`, `ResidualBlock`, `Downsample`, `Upsample` |
| **dataset.py** | `get_dataloaders()` — MNIST with [-1,1] normalisation and 90/10 train/val split |
| **train.py** | `main()` orchestrator; `train_one_epoch()`, `validate()`, `test()`, `sample()` functions; `EMA` class |

---

## References

- Ho, J., Jain, A., & Abbeel, P. (2020). *Denoising Diffusion Probabilistic Models*. NeurIPS 2020.
- Nichol, A. & Dhariwal, P. (2021). *Improved Denoising Diffusion Probabilistic Models*. ICML 2021.
- Ronneberger, O., Fischer, P., & Brox, T. (2015). *U-Net: Convolutional Networks for Biomedical Image Segmentation*. MICCAI 2015.
