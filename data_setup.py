"""
Contains functionality fir creating PyTorch DataLoader's for image classification for image classification data.
"""
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
def create_dataloaders(
        train_dir: str,
        test_dir: str,
        transform: transforms.Compose,
        batch_size: int,
        num_workers: int
):
    """Creates training and testing DataLoaders for image classification

    Takes in a training directory and testing directory path and turns them into PyTorch Datasets and then into PyTorch DataLoaders.

    Args:
        train_dir: Path to the training directory.
        test_dir: Path to the testing directory.
        transform: torchvision transforms to perform on training and testing data.
        batch_size: Number of samples per batch in each of the DataLoaders.
        num_workers: An integer for the number of workers per DataLoader.
    Returns:
        A tuple of (train_dataloader, test_dataloader, class_names).
        Where class_names is a list of the target classes.
        Example usage:
            train_dataloader, test_dataloader, class_names = create_dataloaders(train_dir = path/to/train_dir,
            test_dir = path/to/test_dir, transform = some_transform, batch_size = 32, num_workers = 4)"""
    # Use ImageFolder to create Dataset(s)
    train_data = datasets.ImageFolder(train_dir, transform=transform)
    test_data = datasets.ImageFolder(test_dir, transform= transform)

    # Get class names
    class_names = train_data.classes
    # Turn images into DataLoaders
    train_dataloader = DataLoader(dataset=train_data,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers,
                                  pin_memory=True)
    test_dataloader = DataLoader(dataset=test_data,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=num_workers,
                                 pin_memory=True)
    return train_dataloader, test_dataloader, class_names

# Weighted dataloaders

import torch
from torch.utils.data import WeightedRandomSampler


def _balanced_sample_weights(targets, num_classes: int) -> torch.Tensor:
    """Per-sample weight = 1 / (number of samples in that sample's class)."""
    targets = torch.as_tensor(targets, dtype=torch.long)
    class_counts = torch.bincount(targets, minlength=num_classes).float()
    class_weights = 1.0 / class_counts.clamp(min=1)  # clamp guards empty classes
    return class_weights[targets]


def create_weighted_dataloaders(
        train_dir: str,
        val_dir: str,
        transform: transforms.Compose,
        batch_size: int,
        num_workers: int,
        seed: int = 42
):
    """Creates class-balanced (weighted) training and validation DataLoaders.

    Each sample is drawn with probability proportional to 1 / (size of its class),
    so every class is seen about equally often. The dataset itself is untouched;
    only the order/frequency in which samples are fed to the model changes.

    - Train: a WeightedRandomSampler re-draws a fresh balanced epoch each time
      (len(train_data) samples per epoch, with replacement).
    - Validation: a balanced draw is made ONCE with a fixed seed and reused every
      epoch, so validation curves are comparable across epochs.

    Args:
        train_dir: Path to the training directory.
        val_dir: Path to the validation directory.
        transform: torchvision transforms applied to both sets.
        batch_size: Samples per batch.
        num_workers: Workers per DataLoader.
        seed: Seed for the balanced draws.

    Returns:
        (train_dataloader, val_dataloader, class_names)
    """
    train_data = datasets.ImageFolder(train_dir, transform=transform)
    val_data = datasets.ImageFolder(val_dir, transform=transform)
    class_names = train_data.classes
    num_classes = len(class_names)

    # Train
    train_weights = _balanced_sample_weights(train_data.targets, num_classes)
    train_generator = torch.Generator().manual_seed(seed)
    train_sampler = WeightedRandomSampler(weights=train_weights,
                                          num_samples=len(train_data),
                                          replacement=True,
                                          generator=train_generator)

    # Validation
    val_weights = _balanced_sample_weights(val_data.targets, num_classes)
    val_generator = torch.Generator().manual_seed(seed)
    val_indices = torch.multinomial(val_weights,
                                    num_samples=len(val_data),
                                    replacement=True,
                                    generator=val_generator).tolist()

    # NOTE: when a sampler is given, shuffle must be left at its default (False).
    train_dataloader = DataLoader(dataset=train_data,
                                  batch_size=batch_size,
                                  sampler=train_sampler,
                                  num_workers=num_workers,
                                  pin_memory=True)
    val_dataloader = DataLoader(dataset=val_data,
                                batch_size=batch_size,
                                sampler=val_indices,
                                num_workers=num_workers,
                                pin_memory=True)
    return train_dataloader, val_dataloader, class_names
