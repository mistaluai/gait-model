import os

import torch
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from data.data_preprocessor import load_gait_sequences
from data.dataset import GaitOpticalFlowDataset, GaitFrameSequenceDataset
from models.gaitFlow3DCNN import Flow3DCNNClassifier
from models.gaitFlowLSTM import FlowConvLSTMClassifier
from models.gaitLSTM import GEIConvLSTMClassifier
from utils.random_seed import set_seed
from utils.visualization import visualize_fold_accuracies
from training.training import run_kfold_training, train_model


def main(path: str = None):
    # Path to your dataset folder
    dataset_path = path or "data/binary"

    # Load dataframe
    df = load_gait_sequences(dataset_path, load_images=False)

    # Model and training parameters
    num_classes = len(df['label'].unique())
    k_folds = 5
    epochs = 10
    batch_size = 32
    lr = 1e-3
    num_workers = 2

    # Run k-fold training
    accuracies = run_kfold_training(
        df=df,
        model_class=GEIConvLSTMClassifier,
        num_classes=num_classes,
        k_folds=k_folds,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        num_workers=num_workers,
        use_tqdm=True
    )

    # Visualize results
    visualize_fold_accuracies(accuracies)

def main_flow(
    path: str = None,
    k_folds: int = 5,
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-3,
    num_workers: int = 2,
    use_tqdm: bool = True,
    use_tvl1: bool = True,
    augment_flow=None,
    seed: int = 2005
):
    """
    Train a GEIConvLSTM model using temporal optical flow dataset with k-fold cross-validation.

    Args:
        path (str): Path to dataset directory (default: "data/binary")
        k_folds (int): Number of cross-validation folds
        epochs (int): Training epochs per fold
        batch_size (int): Batch size for training
        lr (float): Learning rate
        num_workers (int): Number of DataLoader workers
        use_tqdm (bool): Whether to show training progress bars
        use_tvl1 (bool): Use TV-L1 optical flow instead of Farneback
        augment_flow (callable): Optional flow augmentation function
        seed (int): Random seed
    """
    # Set seed
    set_seed(seed)

    # Resolve dataset path
    dataset_path = path or "data/binary"

    # Load gait sequence metadata
    df = load_gait_sequences(dataset_path, load_images=False)

    # Determine number of output classes
    num_classes = len(df['label'].unique())

    # Run training loop
    accuracies = run_kfold_training(
        df=df,
        model_class=Flow3DCNNClassifier,
        dataset_class=GaitOpticalFlowDataset,
        num_classes=num_classes,
        k_folds=k_folds,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        num_workers=num_workers,
        use_tqdm=use_tqdm,
        use_tvl1=use_tvl1,
        flow_augment=augment_flow
    )

    # Plot final results
    visualize_fold_accuracies(accuracies)

def main_flow_no_folds(
    path: str = None,
    test_size: float = 0.2,
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-3,
    num_workers: int = 2,
    use_tqdm: bool = True,
    use_tvl1: bool = True,
    augment_flow=None,
    seed: int = 2005
):
    """
    Train a flow-based gait recognition model with a single train/val split (no k-fold).

    Args:
        path (str): Dataset path
        test_size (float): Proportion of validation data
        epochs (int): Training epochs
        batch_size (int): Batch size
        lr (float): Learning rate
        num_workers (int): DataLoader workers
        use_tqdm (bool): Use progress bars
        use_tvl1 (bool): Use TV-L1 optical flow
        augment_flow (callable): Optional flow augmentation
        seed (int): Random seed
    """
    # Set reproducibility
    set_seed(seed)

    # Load metadata
    dataset_path = path or "data/binary"
    df = load_gait_sequences(dataset_path, load_images=False)
    num_classes = len(df["label"].unique())

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=seed)
    train_idx, val_idx = next(splitter.split(df, df["label"]))
    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)

    # Create datasets
    train_dataset = GaitOpticalFlowDataset(
        dataframe=train_df,
        train_augmentations=augment_flow,
        use_tvl1=use_tvl1
    )

    val_dataset = GaitOpticalFlowDataset(
        dataframe=val_df,
        train_augmentations=None,
        use_tvl1=use_tvl1,
        label_to_index=train_dataset.label_to_index  # Ensure same label mapping
    )

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model = Flow3DCNNClassifier(num_classes=num_classes).to(device)

    # Train
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr)

    train_acc, val_acc = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=epochs,
        use_tqdm=use_tqdm
    )

    print(f"\nTrain Accuracy: {train_acc:.4f}")
    print(f"Val Accuracy: {val_acc:.4f}")
