from data.data_preprocessor import load_gait_sequences
from data.dataset import GaitSequenceDataset
from models.gaitLSTM import GEIConvLSTMClassifier
from training.training import run_kfold_training
from utils.visualization import visualize_fold_accuracies

def main():
    # Path to your dataset folder
    dataset_path = "GEI_maps/binary"  # Change as needed

    # Load dataframe
    df = load_gait_sequences(dataset_path, load_images=True)
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

if __name__ == "__main__":
    main()