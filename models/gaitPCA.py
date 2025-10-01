import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt

class PCA:
    '''
    a class to extract the pca of the dataset
    these components used in SVM model then to work with.
    
    The dataset is loaded using the gaitclass of the dataset
    '''

    def __init__(self, num_components=None):
        self.num_components = num_components
        self.mean_face = None
        self.eigenfaces = None
        self.projections = None
        self.components = None

    def fit(self, X):
        """
        X: shape (n_samples, n_features) where each row is a flattened image.
        """
        self.mean_face = np.mean(X, axis=0)
        X_centered = X - self.mean_face
        n_samples, n_features = X_centered.shape
        print(f"Fitting PCA with {n_samples} samples and {n_features} features.")
        print(f"Number of components requested: {self.num_components}")
        print(f"Shape of centered data: {X_centered.shape}")

        # Standard PCA
        cov_matrix = np.cov(X_centered.T)
        eigvals, eigvecs = np.linalg.eigh(cov_matrix)
        sorted_indices = np.argsort(eigvals)[::-1]
        eigvals = eigvals[sorted_indices]
        eigvecs = eigvecs[:, sorted_indices]

        if self.num_components is not None:
            eigvecs = eigvecs[:, :self.num_components]

        eigvecs = eigvecs / np.linalg.norm(eigvecs, axis=0)
        self.eigenfaces = eigvecs
        self.components = eigvecs.T
        self.projections = np.dot(X_centered, self.eigenfaces)

    def transform(self, X):
        """
        Project data into eigenface space.
        """
        X_centered = X - self.mean_face
        return np.dot(X_centered, self.eigenfaces)

    def inverse_transform(self, projections):
        """
        Reconstruct from eigenface projection.
        """
        return np.dot(projections, self.eigenfaces.T) + self.mean_face

    def get_eigenfaces(self):
        return self.eigenfaces

    def get_mean_face(self):
        return self.mean_face
    
    


class Recognizer:
    def __init__(self, num_components=50, method="cosine"):
        self.pca = PCA(num_components=num_components)
        self.method = method
        self.svm = None
        self.y_train = None
        self.train_projections = None
        self.train_labels = None
        self.train_data = None

    def train(self, X_train, y_train):
        """Train the face recognizer"""
        self.y_train = y_train
        self.train_data = X_train
        self.pca.fit(X_train)
        X_train_proj = self.pca.transform(X_train)
        self.train_projections = X_train_proj
        self.train_labels = y_train
        if self.method == "svm":
            self.svm = SVC(kernel='linear', probability=True)
            self.svm.fit(X_train_proj, y_train)

    def predict(self, face_vector, threshold=None):
        """
        Predict the identity of a given face using cosine similarity.

        Parameters:
            face_vector (ndarray): Flattened input face image.
            threshold (float, optional): Cosine similarity threshold ∈ [0, 1] for rejecting unknowns.

        Returns:
            best_match_face (ndarray or None): Closest training face (or None if rejected).
            best_label (str): Predicted label or 'unknown'.
            confidence (float): Cosine similarity ∈ [0, 1].
        """
        face_projected = self.pca.transform(face_vector.reshape(1, -1))[0]
        if self.method == "svm":
            pred_label = self.svm.predict(face_projected.reshape(1, -1))[0]
            confidence = np.max(self.svm.predict_proba(face_projected.reshape(1, -1)))
            # Optionally use threshold for "unknown"
            # if threshold is not None and confidence < threshold:
            #     return None, "unknown", confidence
            return None, pred_label, confidence
        else:
            # Cosine similarity method
            def cosine_sim(a, b):
                return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)
            similarities = np.array([
                cosine_sim(face_projected, train_vec)
                for train_vec in self.train_projections
            ])
            best_index = np.argmax(similarities)
            max_similarity = similarities[best_index]
            if threshold is not None and max_similarity < threshold:
                return None, "unknown", max_similarity
            best_label = self.train_labels[best_index]
            best_match_face = self.train_data[best_index]
            return best_match_face, best_label, max_similarity

    def evaluate(self, X_test, y_test, threshold=0.5, include_unknown=False):
        num_correct = 0
        num_unknown = 0
        total = len(X_test)
        y_true = []
        y_pred = []
        decision_scores = []

        for i in range(total):
            _, pred_label, pred_score = self.predict(X_test[i], threshold=threshold)
            if pred_label == "unknown":
                num_unknown += 1
                if include_unknown:
                    continue
            else:
                y_true.append(y_test[i])
                y_pred.append(pred_label)
                decision_scores.append(pred_score)
                if pred_label == y_test[i]:
                    num_correct += 1
        print(f"Total samples: {total}, Correct: {num_correct}, Unknown: {num_unknown}")
        num_recognized = total - num_unknown
        accuracy = num_correct / total if include_unknown else (num_correct / num_recognized if num_recognized > 0 else 0.0)
        rejection_rate = num_unknown / total
        confusion = confusion_matrix(y_true, y_pred) if y_true and y_pred else None

        # Compute ROC Curve (One-vs-Rest)
        lb = LabelBinarizer()
        print(len(set(y_test)))
        
        if len(set(y_test)) <= 2:
            # Not enough classes to compute ROC
            roc_auc = {}
            fpr = {}
            tpr = {}
        else:
            y_true_bin = lb.fit_transform(y_true)
            fpr, tpr, roc_auc = {}, {}, {}
            for i in range(len(lb.classes_)):
                if y_true_bin[:, i].sum() == 0:  # Avoid ROC calc if no samples for class
                    continue
                fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], np.array(decision_scores))
                roc_auc[i] = auc(fpr[i], tpr[i])

        return {
            'accuracy': accuracy,
            'rejection_rate': rejection_rate,
            'num_correct': num_correct,
            'num_unknown': num_unknown,
            'num_recognized': num_recognized,
            'total': total,
            'confusion_matrix': confusion,
            'roc_curve': roc_auc,
            'fpr': fpr,
            'tpr': tpr,
        }
    def reconstruct(self, X):
        """
        Reconstruct faces from input batch.
        """
        return self.pca.inverse_transform(self.pca.transform(X))

    def get_mean_face(self):
        return self.pca.get_mean_face()

    def get_eigenfaces(self):
        return self.pca.get_eigenfaces()
