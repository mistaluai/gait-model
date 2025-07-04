import numpy as np
from torch.utils.data import Subset, DataLoader

from data.data_preprocessor import load_gait_sequences
from data.dataset import GaitSequenceDataset


def get_fold_indices(dataset_size, k):
    """
    Generate train/validation indices for k-fold CV.

    Args:
        dataset_size (int): Total number of samples.
        k (int): Number of folds.

    Returns:
        List of (train_indices, val_indices) tuples
    """
    indices = np.arange(dataset_size)
    fold_sizes = np.full(k, dataset_size // k, dtype=int)
    fold_sizes[:dataset_size % k] += 1
    current = 0
    folds = []

    for fold_size in fold_sizes:
        val_idx = indices[current:current + fold_size]
        train_idx = np.concatenate((indices[:current], indices[current + fold_size:]))
        folds.append((train_idx, val_idx))
        current += fold_size

    return folds


def get_dataloaders_for_fold(dataset, train_idx, val_idx, batch_size=1, num_workers=1):
    """
    Create DataLoaders for a given fold.

    Args:
        dataset (Dataset): The full dataset.
        train_idx (array-like): Training indices.
        val_idx (array-like): Validation indices.
        batch_size (int): Batch size.
        num_workers (int): Number of workers for loading.

    Returns:
        train_loader, val_loader
    """
    train_subset = Subset(dataset, train_idx)
    val_subset = Subset(dataset, val_idx)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader


def run_kfold_cross_validation(dataset, k=5, batch_size=1, num_workers=1):
    """
    Perform k-fold cross-validation on a PyTorch dataset.

    Args:
        dataset (Dataset): The full dataset.
        k (int): Number of folds.
        batch_size (int): Batch size.
        num_workers (int): DataLoader workers.

    Yields:
        fold_index, train_loader, val_loader
    """
    folds = get_fold_indices(len(dataset), k)

    for i, (train_idx, val_idx) in enumerate(folds):
        print(f"\nProcessing Fold {i + 1}/{k}")
        train_loader, val_loader = get_dataloaders_for_fold(
            dataset, train_idx, val_idx, batch_size, num_workers
        )
        yield i, train_loader, val_loader

if __name__ == "__main__":
    df = load_gait_sequences("./gei_maps/Multiclass6", load_images=False)
    dataset = GaitSequenceDataset(df)

    for fold_idx, train_loader, val_loader in run_kfold_cross_validation(dataset, k=5, batch_size=4):
        # Training
        print(
            f"Fold {fold_idx + 1} has {len(train_loader.dataset)} training samples and {len(val_loader.dataset)} validation samples.")