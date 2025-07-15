from data.data_preprocessor import load_gait_sequences
from data.dataset import GaitSequenceDataset
from models.gaitLSTM import GEIConvLSTMClassifier
from models.optical_flow_CNN import FlowCNN
from training.training import run_kfold_training
from utils.visualization import visualize_fold_accuracies

def main(path: str = None):
    # Path to your dataset folder
    dataset_path = path or "data/binary"

    # Load dataframe
    df = load_gait_sequences(dataset_path, load_images=False)
    dataset = GaitSequenceDataset(df)
    
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
    use_tqdm: bool = True
):
    """
    Train FlowCNN model using accumulated optical flow dataset with k-fold cross-validation.

    Args:
        path (str): Path to the dataset directory.
        k_folds (int): Number of cross-validation folds.
        epochs (int): Number of training epochs per fold.
        batch_size (int): Batch size for training.
        lr (float): Learning rate.
        num_workers (int): Number of data loading workers.
        use_tqdm (bool): Whether to use progress bars.
    """
    # Dataset path
    dataset_path = path or "data/binary"

    # Load sequences
    df = load_gait_sequences(dataset_path, load_images=False)

    # Load dataset with accumulated optical flow
    dataset = GaitSequenceDataset(
        df,
        return_one=True,
        return_metadata=False
    )

    # Get number of classes from dataset
    num_classes = len(df['label'].unique())

    # Run training using FlowCNN
    accuracies = run_kfold_training(
        dataset=dataset,
        model_class=FlowCNN,
        num_classes=num_classes,
        k_folds=k_folds,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        num_workers=num_workers,
        use_tqdm=use_tqdm
    )

    # Show results
    visualize_fold_accuracies(accuracies)

if __name__ == "__main__":
    main("data/binary")