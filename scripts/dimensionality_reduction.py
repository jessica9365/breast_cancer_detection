import numpy as np
import os
from pathlib import Path
from sklearn.decomposition import PCA
import joblib


def load_processed_split(split_dir):
    """
    Loads processed image and label arrays from split directory.
    Args:
        split_dir (str): Directory containing X.npy and y.npy
    Returns:
        X (np.ndarray): Image array (N, 128, 128)
        y (np.ndarray): Labels array (N,)
    """
    X = np.load(os.path.join(split_dir, 'X.npy'))
    y = np.load(os.path.join(split_dir, 'y.npy'))
    return X, y

def flatten_images(X):
    """
    Flattens each image in array to 1D vector.
    Args:
        X (np.ndarray): (N, H, W) array of images
    Returns:
        X_flat (np.ndarray): (N, H*W) array
    """
    n_samples = X.shape[0]
    return X.reshape(n_samples, -1)

def apply_pca(X_train_flat, X_val_flat, X_test_flat, n_components=50):
    """
    Fits PCA on X_train_flat and transforms train, val, and test sets to reduced dimensions.
    Args:
        X_train_flat (ndarray): Training data, shape (N_train, n_features)
        X_val_flat (ndarray): Validation data, shape (N_val, n_features)
        X_test_flat (ndarray): Test data, shape (N_test, n_features)
        n_components (int): Number of dimensions to keep
    Returns:
        X_train_pca, X_val_pca, X_test_pca
    """
    pca = PCA(n_components=n_components, random_state=42)
    X_train_pca = pca.fit_transform(X_train_flat)
    X_val_pca = pca.transform(X_val_flat)
    X_test_pca = pca.transform(X_test_flat)
    return X_train_pca, X_val_pca, X_test_pca, pca

def main():
    data_root = "/Users/jessica/Documents/GitHub/breast_cancer_detection/data/processed"
    save_root = data_root  # Saving under processed dir; can change as needed

    train_dir = os.path.join(data_root, 'train')
    val_dir = os.path.join(data_root, 'val')
    test_dir = os.path.join(data_root, 'test')

    print("Loading processed splits...")
    X_train, y_train = load_processed_split(train_dir)
    X_val, y_val = load_processed_split(val_dir)
    X_test, y_test = load_processed_split(test_dir)

    print("Flattening images...")
    X_train_flat = flatten_images(X_train)
    X_val_flat = flatten_images(X_val)
    X_test_flat = flatten_images(X_test)

    n_components = 140  # Determined from exploratory notebook
    print(f"Applying PCA with n_components={n_components}...")
    X_train_pca, X_val_pca, X_test_pca, pca = apply_pca(X_train_flat, X_val_flat, X_test_flat, n_components=n_components)

    print("Saving PCA-transformed data and PCA model...")
    # Save PCA features for each split
    np.save(os.path.join(train_dir, 'X_pca.npy'), X_train_pca)
    np.save(os.path.join(val_dir, 'X_pca.npy'), X_val_pca)
    np.save(os.path.join(test_dir, 'X_pca.npy'), X_test_pca)
    # Save PCA object for future use
    joblib.dump(pca, os.path.join(save_root, 'pca_model_140.pkl'))
    print("Done.")

if __name__ == "__main__":
    main()

