"""
Training, validation, testing, and sampling for DDPM on MNIST.

Training objective (simplified DDPM loss):
    L_simple = E_{t, x_0, ε} [ ‖ ε − ε_θ(x_t, t) ‖² ]

Sampling (reverse process):
    x_{t-1} = (1/√α_t) · ( x_t − (β_t / √(1-ᾱ_t)) · ε_θ(x_t, t) ) + σ_t · z
    where z ~ N(0, I) for t > 1, and z = 0 for t = 1.
"""

import os
import copy
import math
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from tqdm import tqdm

import config
from diffusion import DiffusionSchedule
from model import UNet
from dataset import get_dataloaders


# ─── Exponential Moving Average ────────────────────────────────────────────────

class EMA:
    """Track an exponential moving average of model parameters."""

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow = copy.deepcopy(model)
        self.shadow.eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module):
        for s_param, m_param in zip(self.shadow.parameters(), model.parameters()):
            s_param.data.mul_(self.decay).add_(m_param.data, alpha=1 - self.decay)

    def forward(self, *args, **kwargs):
        return self.shadow(*args, **kwargs)


# ─── Sampling ──────────────────────────────────────────────────────────────────

@torch.no_grad()
def sample(model: nn.Module, schedule: DiffusionSchedule,
           n_samples: int = 64, device: str = "cpu"):
    """
    Generate images via the full DDPM reverse process.

    Algorithm 2 from the paper:
        for t = T, T-1, …, 1:
            z ~ N(0, I) if t > 1 else z = 0
            x_{t-1} = (1/√α_t)(x_t − (β_t/√(1-ᾱ_t)) ε_θ(x_t,t)) + σ_t z
    """
    model.eval()
    img = torch.randn(n_samples, config.IN_CHANNELS,
                       config.IMAGE_SIZE, config.IMAGE_SIZE, device=device)

    for t in tqdm(reversed(range(schedule.T)), total=schedule.T, desc="Sampling"):
        t_batch = torch.full((n_samples,), t, device=device, dtype=torch.long)
        predicted_noise = model(img, t_batch)

        alpha = schedule.alphas[t]
        alpha_bar = schedule.alphas_cumprod[t]
        beta = schedule.betas[t]

        # Mean of p_θ(x_{t-1} | x_t)
        mean = (1.0 / torch.sqrt(alpha)) * (
            img - (beta / torch.sqrt(1.0 - alpha_bar)) * predicted_noise
        )

        if t > 0:
            sigma = torch.sqrt(beta)
            noise = torch.randn_like(img)
            img = mean + sigma * noise
        else:
            img = mean

    # Clamp to [-1, 1]
    img = torch.clamp(img, -1.0, 1.0)
    return img


# ─── Training loop ─────────────────────────────────────────────────────────────

def train_one_epoch(model, optimizer, schedule, dataloader, device,
                    ema=None, scheduler=None):
    model.train()
    total_loss = 0.0
    count = 0

    for batch_images, _ in dataloader:
        batch_images = batch_images.to(device)
        B = batch_images.shape[0]

        # Sample random time-steps uniformly: t ~ Uniform({0, …, T-1})
        t = torch.randint(0, schedule.T, (B,), device=device, dtype=torch.long)

        # Sample noise ε ~ N(0, I)
        noise = torch.randn_like(batch_images)

        # Forward diffusion: x_t = √ᾱ_t x_0 + √(1-ᾱ_t) ε
        x_t = schedule.q_sample(batch_images, t, noise)

        # Predict the noise
        predicted_noise = model(x_t, t)

        # Simple MSE loss: ‖ε − ε_θ(x_t, t)‖²
        loss = nn.functional.mse_loss(predicted_noise, noise)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
        optimizer.step()

        # ── Per-step EMA update (critical for sample quality!) ──────────
        if ema is not None:
            ema.update(model)
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item() * B
        count += B

    return total_loss / count


@torch.no_grad()
def validate(model, schedule, dataloader, device):
    model.eval()
    total_loss = 0.0
    count = 0

    for batch_images, _ in dataloader:
        batch_images = batch_images.to(device)
        B = batch_images.shape[0]
        t = torch.randint(0, schedule.T, (B,), device=device, dtype=torch.long)
        noise = torch.randn_like(batch_images)
        x_t = schedule.q_sample(batch_images, t, noise)
        predicted_noise = model(x_t, t)
        loss = nn.functional.mse_loss(predicted_noise, noise)
        total_loss += loss.item() * B
        count += B

    return total_loss / count


@torch.no_grad()
def test(model, schedule, dataloader, device):
    """Run full test-set evaluation and return average MSE loss."""
    model.eval()
    total_loss = 0.0
    count = 0

    for batch_images, _ in dataloader:
        batch_images = batch_images.to(device)
        B = batch_images.shape[0]
        t = torch.randint(0, schedule.T, (B,), device=device, dtype=torch.long)
        noise = torch.randn_like(batch_images)
        x_t = schedule.q_sample(batch_images, t, noise)
        predicted_noise = model(x_t, t)
        loss = nn.functional.mse_loss(predicted_noise, noise)
        total_loss += loss.item() * B
        count += B

    return total_loss / count


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    # ── Setup ───────────────────────────────────────────────────────────────
    torch.manual_seed(config.SEED)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(config.SAMPLE_DIR, exist_ok=True)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    writer = SummaryWriter(log_dir=config.LOG_DIR)

    # ── Data ────────────────────────────────────────────────────────────────
    train_loader, val_loader, test_loader = get_dataloaders()
    print(f"Train batches: {len(train_loader)}, "
          f"Val batches: {len(val_loader)}, "
          f"Test batches: {len(test_loader)}")

    # ── Model ───────────────────────────────────────────────────────────────
    model = UNet(
        in_channels=config.IN_CHANNELS,
        model_channels=config.MODEL_CHANNELS,
        channel_mult=config.CHANNEL_MULT,
        num_res_blocks=config.NUM_RES_BLOCKS,
        time_emb_dim=config.TIME_EMB_DIM,
        dropout=config.DROPOUT,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # ── Optimiser & schedule ────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
    ema = EMA(model, decay=config.EMA_DECAY)

    # ── LR scheduler: linear warmup → cosine decay (stepped per batch) ────
    total_steps = config.NUM_EPOCHS * len(train_loader)
    warmup_steps = getattr(config, "LR_WARMUP_STEPS", 500)

    def _lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return max(0.05, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)

    schedule = DiffusionSchedule(
        T=config.T,
        beta_start=config.BETA_START,
        beta_end=config.BETA_END,
        schedule=config.BETA_SCHEDULE,
        device=device,
    )

    # ── Log a batch of real images for reference ────────────────────────────
    real_batch, _ = next(iter(train_loader))
    real_grid = make_grid(real_batch[:64], nrow=8, normalize=True, value_range=(-1, 1))
    writer.add_image("Real_images", real_grid, 0)

    # ── Log noising process visualisation ───────────────────────────────────
    _log_forward_process(writer, real_batch[:8], schedule, device)

    # ── Training ────────────────────────────────────────────────────────────
    best_val_loss = float("inf")

    for epoch in range(1, config.NUM_EPOCHS + 1):
        train_loss = train_one_epoch(model, optimizer, schedule,
                                      train_loader, device,
                                      ema=ema, scheduler=scheduler)

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("LR", optimizer.param_groups[0]["lr"], epoch)
        print(f"Epoch {epoch:3d}/{config.NUM_EPOCHS}  train_loss={train_loss:.6f}", end="")

        # Validation (using EMA model for consistency with sampling)
        if epoch % config.VALIDATE_EVERY == 0:
            val_loss = validate(ema.shadow, schedule, val_loader, device)
            writer.add_scalar("Loss/val", val_loss, epoch)
            print(f"  val_loss={val_loss:.6f}", end="")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                _save_checkpoint(model, optimizer, ema, epoch, val_loss,
                                 os.path.join(config.CHECKPOINT_DIR, "best.pt"))
                print("  ★ best", end="")

        print()

        # Sampling
        if epoch % config.SAMPLE_EVERY == 0 or epoch == 1:
            samples = sample(ema.shadow, schedule, config.NUM_SAMPLES, device)
            grid = make_grid(samples, nrow=8, normalize=True, value_range=(-1, 1))
            writer.add_image("Samples/ema", grid, epoch)

            # Save to disk
            from torchvision.utils import save_image
            save_image(grid, os.path.join(config.SAMPLE_DIR, f"epoch_{epoch:03d}.png"))

    # ── Test ────────────────────────────────────────────────────────────────
    print("\n── Testing ──")
    test_loss = test(ema.shadow, schedule, test_loader, device)
    writer.add_scalar("Loss/test", test_loss, config.NUM_EPOCHS)
    print(f"Test loss (EMA model): {test_loss:.6f}")

    # ── Final samples ───────────────────────────────────────────────────────
    final_samples = sample(ema.shadow, schedule, config.NUM_SAMPLES, device)
    grid = make_grid(final_samples, nrow=8, normalize=True, value_range=(-1, 1))
    writer.add_image("Samples/final", grid, config.NUM_EPOCHS)
    from torchvision.utils import save_image
    save_image(grid, os.path.join(config.RESULTS_DIR, "final_samples.png"))

    # ── Save final checkpoint ───────────────────────────────────────────────
    _save_checkpoint(model, optimizer, ema, config.NUM_EPOCHS, test_loss,
                     os.path.join(config.CHECKPOINT_DIR, "final.pt"))

    writer.close()
    print("Done! Check TensorBoard for training curves and generated samples.")


# ─── Helpers ───────────────────────────────────────────────────────────────────

def _save_checkpoint(model, optimizer, ema, epoch, loss, path):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "ema_state_dict": ema.shadow.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }, path)


def _log_forward_process(writer, images, schedule, device):
    """Log the forward noising process at several time-steps."""
    import torchvision.utils as vutils
    images = images.to(device)
    steps = [0, 50, 100, 200, 500, 999]
    all_imgs = []
    for t_val in steps:
        t = torch.full((images.shape[0],), t_val, device=device, dtype=torch.long)
        noisy = schedule.q_sample(images, t)
        all_imgs.append(noisy)
    grid = vutils.make_grid(torch.cat(all_imgs, dim=0), nrow=images.shape[0],
                            normalize=True, value_range=(-1, 1))
    writer.add_image("Forward_process/noising", grid, 0)


if __name__ == "__main__":
    main()
