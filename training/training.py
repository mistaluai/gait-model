import torch
from sklearn.metrics import accuracy_score
import numpy as np
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm

from data.data_preprocessor import load_gait_sequences
from data.dataset import GaitSequenceDataset
from data.kcv import run_kfold_cross_validation

import inspect

from models.gaitLSTM import GEIConvLSTMClassifier
from utils.visualization import visualize_fold_accuracies


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    scheduler=None,
    num_epochs=10,
    use_tqdm=True
):
    """
    Generic training loop for models with/without sequence lengths (LSTM or CNN).
    Automatically checks if the model's forward method expects `seq_lengths`.

    Args:
        model (nn.Module): Model to train.
        train_loader (DataLoader): Training data.
        val_loader (DataLoader): Validation data.
        criterion: Loss function.
        optimizer: Optimizer.
        device: Torch device.
        scheduler: Learning rate scheduler.
        num_epochs: Number of epochs.
        use_tqdm (bool): Whether to show tqdm progress bars.

    Returns:
        float: Validation accuracy.
    """
    model = model.to(device)
    use_seq_len = "seq_lengths" in inspect.signature(model.forward).parameters

    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0

        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}", leave=False) if use_tqdm else train_loader

        for x, y, *extras in loop:
            x, y = x.to(device), y.to(device)
            seq_lengths = extras[0].to(device) if use_seq_len and extras else None

            optimizer.zero_grad()
            outputs = model(x, seq_lengths) if use_seq_len and seq_lengths is not None else model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        if scheduler:
            scheduler.step()

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch}/{num_epochs} - Training Loss: {avg_loss:.4f}")

    # Evaluation
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        loop = tqdm(val_loader, desc="Evaluating", leave=False) if use_tqdm else val_loader
        for x, y, *extras in loop:
            x = x.to(device)
            seq_lengths = extras[0].to(device) if use_seq_len and extras else None

            outputs = model(x, seq_lengths) if use_seq_len and seq_lengths is not None else model(x)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.numpy())

    acc = accuracy_score(all_labels, all_preds)
    print(f"Validation Accuracy: {acc:.4f}")
    return acc


def run_kfold_training(
    dataset,
    model_class,
    num_classes,
    k_folds=5,
    epochs=20,
    batch_size=4,
    lr=1e-3,
    num_workers=1,
    use_tqdm=True
):
    """
    Runs k-fold cross-validation for any model (CNN or LSTM).

    Args:
        dataset (Dataset): GaitSequenceDataset.
        model_class (nn.Module): Model class.
        num_classes (int): Number of classes.
        k_folds (int): Folds count.
        epochs (int): Epochs per fold.
        batch_size (int): Batch size.
        lr (float): Learning rate.
        num_workers (int): DataLoader workers.
        use_tqdm (bool): Whether to use tqdm progress bars.

    Returns:
        List[float]: Accuracy for each fold.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    fold_accuracies = []

    for fold_idx, train_loader, val_loader in run_kfold_cross_validation(
        dataset, k=k_folds, batch_size=batch_size, num_workers=num_workers
    ):
        print(f"\n--- Fold {fold_idx + 1} ---")

        model = model_class(num_classes=num_classes).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = Adam(model.parameters(), lr=lr)

        acc = train_model(
            model,
            train_loader,
            val_loader,
            criterion,
            optimizer,
            device,
            num_epochs=epochs,
            use_tqdm=use_tqdm
        )
        fold_accuracies.append(acc)

    print(f"\nAverage Accuracy over {k_folds} folds: {np.mean(fold_accuracies):.4f}")
    return fold_accuracies


if __name__ == "__main__":
    df = load_gait_sequences("/Users/mistaluai/Documents/Github Repos/gait-model/data/gei_maps/Multiclass6", load_images=False)
    dataset = GaitSequenceDataset(df)

    accuracies = run_kfold_training(
        dataset=dataset,
        model_class=GEIConvLSTMClassifier,
        num_classes=6,
        k_folds=5,
        epochs=10,
        batch_size=32,
        lr=1e-3,
        num_workers=2,
        use_tqdm=True
    )

    visualize_fold_accuracies(accuracies)