from data.data_preprocessor import load_gait_frames
from models.gaitPCA import PCA, Recognizer
from sklearn.model_selection import train_test_split
from PIL import Image

import numpy as np

def main(path: str = None, num_components: int = 5, test_size: float = 0.3, random_state: int = 42):
    dataset_path = path or "GEI_maps/binary"

    # Load each frame as a separate row with its label
    df = load_gait_frames(dataset_path, load_images=True)

    images = [img for img in df['image'].values]  # shape: (n_samples, H, W)
    
    # Downscale images to reduce memory usage (e.g., 64x64)
    target_size = (64, 64)
    images_resized = [img.resize(target_size, Image.BILINEAR) for img in images]
    X = np.stack([np.array(img) for img in images_resized])
    print("images shape after resize:", X.shape)

    X = X.reshape(X.shape[0], -1)  # Flatten images
    y = df['label'].values

    # Print dataset information
    print("Total samples:", len(df))
    print("Class distribution:\n", df['label'].value_counts())

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # Use the Recognizer class for PCA-based recognition
    recognizer = Recognizer(num_components=num_components, method='svm')
    recognizer.train(X_train, y_train)
    print(len(y_test), "samples in test set")
    # Evaluate on test set
    eval_results = recognizer.evaluate(X_test, y_test, threshold=None, include_unknown=False)

    print(f"PCA+Recognizer Test Accuracy: {eval_results['accuracy']:.4f}")

if __name__ == "__main__":
    main("GEI_maps/Multiclass6", num_components=20, test_size=0.3, random_state=42)