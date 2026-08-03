"""
Configuration for DDPM on MNIST.
Based on: "Denoising Diffusion Probabilistic Models" (Ho et al., 2020)
"""

import os

# ─── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
LOG_DIR = os.path.join(PROJECT_DIR, "runs")
CHECKPOINT_DIR = os.path.join(PROJECT_DIR, "checkpoints")
SAMPLE_DIR = os.path.join(PROJECT_DIR, "samples")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")

# ─── Diffusion hyper-parameters ────────────────────────────────────────────────
T = 1000                    # total diffusion time-steps
BETA_START = 1e-4           # beta_1
BETA_END = 0.02             # beta_T
BETA_SCHEDULE = "cosine"    # "linear" or "cosine"  (cosine → better quality)

# ─── Model hyper-parameters ────────────────────────────────────────────────────
IMAGE_SIZE = 28             # MNIST is 28x28
IN_CHANNELS = 1             # grayscale
MODEL_CHANNELS = 128        # base channel width (larger → better quality)
CHANNEL_MULT = (1, 2, 4)   # channel multiplier at each resolution
NUM_RES_BLOCKS = 2          # residual blocks per resolution level
TIME_EMB_DIM = 256          # learned time-embedding dimension
DROPOUT = 0.1               # dropout rate

# ─── Training hyper-parameters ─────────────────────────────────────────────────
BATCH_SIZE = 128
LEARNING_RATE = 2e-4
NUM_EPOCHS = 50
EMA_DECAY = 0.99          # exponential moving average decay
GRAD_CLIP = 1.0             # gradient clipping norm
LR_WARMUP_STEPS = 5       # linear warmup before cosine decay
NUM_WORKERS = 4
SEED = 42

# ─── Sampling / evaluation ─────────────────────────────────────────────────────
NUM_SAMPLES = 64            # images to generate for visual inspection
SAMPLE_EVERY = 10           # sample every N epochs
VALIDATE_EVERY = 1          # validate every N epochs

# ─── Device ────────────────────────────────────────────────────────────────────
DEVICE = "cuda"             # will fall back to cpu at runtime if needed
