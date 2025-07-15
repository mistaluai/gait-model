import torch
from sklearn.metrics import accuracy_score
import numpy as np
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
import inspect

from data.data_preprocessor import load_gait_sequences
from data.dataset import GaitFrameSequenceDataset
from models.gaitLSTM import GEIConvLSTMClassifier
from utils.visualization import visualize_fold_accuracies
from data.kcv import run_kfold_cross_validation


def evaluate_model(model, data_loader, device, use_seq_len, use_tqdm=True, label="Validation"):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        loop = tqdm(data_loader, desc=f"Evaluating {label}", leave=False) if use_tqdm else data_loader
        for x, y, *extras in loop:
            x = x.to(device)
            y = y.to(device)
            seq_lengths = extras[0].to(device) if use_seq_len and extras else None

            outputs = model(x, seq_lengths) if use_seq_len and seq_lengths is not None else model(x)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    print(f"{label} Accuracy: {acc:.4f}")
    return acc


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

    # Evaluate on training and validation sets
    train_acc = evaluate_model(model, train_loader, device, use_seq_len, use_tqdm, label="Training")
    val_acc = evaluate_model(model, val_loader, device, use_seq_len, use_tqdm, label="Validation")

    return train_acc, val_acc


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
    device = torch.device("cuda" if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    train_accuracies = []
    val_accuracies = []

    for fold_idx, train_loader, val_loader in run_kfold_cross_validation(
        dataset, k=k_folds, batch_size=batch_size, num_workers=num_workers
    ):
        print(f"\n--- Fold {fold_idx + 1} ---")

        model = model_class(num_classes=num_classes).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = Adam(model.parameters(), lr=lr)

        train_acc, val_acc = train_model(
            model,
            train_loader,
            val_loader,
            criterion,
            optimizer,
            device,
            num_epochs=epochs,
            use_tqdm=use_tqdm
        )

        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

    print(f"\nAverage Training Accuracy: {np.mean(train_accuracies):.4f}")
    print(f"Average Validation Accuracy: {np.mean(val_accuracies):.4f}")

    return val_accuracies


if __name__ == "__main__":
    df = load_gait_sequences("/Users/mistaluai/Documents/Github Repos/gait-model/data/gei_maps/Multiclass6", load_images=False)
    dataset = GaitFrameSequenceDataset(df)

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