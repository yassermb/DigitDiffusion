"""
MNIST data loading with train / validation / test splits.
"""

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

import config


def get_dataloaders(batch_size: int = None, num_workers: int = None,
                    val_fraction: float = 0.1):
    """
    Returns (train_loader, val_loader, test_loader) for MNIST.

    • Images are scaled to [-1, 1] (standard for diffusion models).
    • A fraction of the training set is held out for validation.
    """
    batch_size = batch_size or config.BATCH_SIZE
    num_workers = num_workers or config.NUM_WORKERS

    transform = transforms.Compose([
        transforms.ToTensor(),                      # [0, 1]
        transforms.Normalize((0.5,), (0.5,)),       # [-1, 1]
    ])

    # Download / load
    full_train = datasets.MNIST(config.DATA_DIR, train=True,
                                download=True, transform=transform)
    test_set = datasets.MNIST(config.DATA_DIR, train=False,
                              download=True, transform=transform)

    # Split train → train + val
    val_size = int(len(full_train) * val_fraction)
    train_size = len(full_train) - val_size
    generator = torch.Generator().manual_seed(config.SEED)
    train_set, val_set = random_split(full_train, [train_size, val_size],
                                      generator=generator)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True,
                              drop_last=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader
