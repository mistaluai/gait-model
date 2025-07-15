from data.data_preprocessor import load_gait_sequences
from models.gaitPCA import PCA
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np

def main(path: str = None, num_components: int = 50, test_size: float = 0.3, random_state: int = 42):
    # Path to your dataset folder
    dataset_path = path or "GEI_maps/binary"

    # Load dataframe
    df = load_gait_sequences(dataset_path, load_images=True)  # 'sequence' column contains lists of images

    # Use the first image of each sequence as a representative
    images = [seq[0] for seq in df['sequence'].values]  # shape: (n_samples, H, W)
    X = np.stack([np.array(img) for img in images])     # Convert PIL images to numpy arrays
    X = X.reshape(X.shape[0], -1)                      # Flatten images
    y = df['label'].values

    # Print dataset information
    print("Total samples:", len(df))
    print("Class distribution:\n", df['label'].value_counts())

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # Fit PCA
    pca = PCA(num_components=num_components)
    pca.fit(X_train)
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)

    # Train SVM
    svm = SVC(kernel='linear', probability=True, random_state=random_state)
    svm.fit(X_train_pca, y_train)

    # Predict
    y_pred = svm.predict(X_test_pca)
    acc = accuracy_score(y_test, y_pred)
    print(f"PCA+SVM Test Accuracy: {acc:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

if __name__ == "__main__":
    main("GEI_maps/binary")