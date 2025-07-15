import os
from data.data_preprocessor import load_gait_sequences
from data.dataset import GaitOpticalFlowDataset, GaitFrameSequenceDataset
from models.gaitLSTM import GEIConvLSTMClassifier
from utils.random_seed import set_seed
from utils.visualization import visualize_fold_accuracies
from training.training import run_kfold_training

def main(path: str = None):
    # Path to your dataset folder
    dataset_path = path or "data/binary"

    # Load dataframe
    df = load_gait_sequences(dataset_path, load_images=False)
    dataset = GaitFrameSequenceDataset(df)
    
    # Model and training parameters
    num_classes = len(df['label'].unique())
    k_folds = 5
    epochs = 10
    batch_size = 32
    lr = 1e-3
    num_workers = 2

    # Run k-fold training
    accuracies = run_kfold_training(
        dataset=dataset,
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

    # Create dataset
    dataset = GaitOpticalFlowDataset(
        dataframe=df,
        return_metadata=False,
        flow_augment=augment_flow,
        use_tvl1=use_tvl1
    )

    # Determine number of output classes
    num_classes = len(df['label'].unique())

    # Run training loop
    accuracies = run_kfold_training(
        dataset=dataset,
        model_class=GEIConvLSTMClassifier,
        num_classes=num_classes,
        k_folds=k_folds,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        num_workers=num_workers,
        use_tqdm=use_tqdm
    )

    # Plot final results
    visualize_fold_accuracies(accuracies)