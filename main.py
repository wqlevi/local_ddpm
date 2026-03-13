from __future__ import annotations

import math

import torch
from torch import optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from trainer import DDPMTrainer, TrainerConfig
from unet import UNet


def build_subset(dataset: datasets.CIFAR10, fraction: float) -> Subset:
    subset_size = max(1, math.floor(len(dataset) * fraction))
    indices = list(range(subset_size))
    return Subset(dataset, indices)


def main() -> None:
    if not (torch.backends.mps.is_available() or torch.cuda.is_available()):
        raise RuntimeError(
            "CUDA or MPS device is required but not available on this machine."
        )

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    train_dataset = datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=transform,
    )
    val_dataset = datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=transform,
    )

    train_subset = build_subset(train_dataset, fraction=0.1)
    val_subset = build_subset(val_dataset, fraction=0.1)

    train_loader = DataLoader(
        train_subset,
        batch_size=64,
        shuffle=True,
        num_workers=2,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=64,
        shuffle=False,
        num_workers=2,
        pin_memory=False,
    )

    model = UNet(in_channels=3, out_channels=3, base_channels=64, t_emb_dim=256)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    config = TrainerConfig(
        timesteps=1000,
        log_every=500,
        eval_every=500,
        num_eval_samples=4,
        output_dir="outputs/cifar10_ddpm",
    )

    trainer = DDPMTrainer(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
    )
    trainer.train(epochs=1000)


if __name__ == "__main__":
    main()
