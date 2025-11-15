import os
import numpy as np
import joblib
import argparse
from sklearn.decomposition import PCA
import json

def load_processed_split(split_dir):
    """
    Loads processed image and label arrays from split directory.
    Args:
        split_dir (str): Directory containing X.npy and y.npy
    Returns:
        X (np.ndarray): Image/feature array (N, features)
        y (np.ndarray): Label array (N,)
    """
    X = np.load(os.path.join(split_dir, 'X.npy'))
    y = np.load(os.path.join(split_dir, 'y.npy'))
    return X, y

def load_config(config_path):
    """
    Load config parameters (data paths, etc.) from JSON file.
    Args:
        config_path (str): Where to read config file.
    Returns:
        config (dict): Dictionary of config parameters.
    """
    with open(config_path, "r") as f:
        config = json.load(f)
    return config

def flatten_images(X):
    """
    Flattens feature or image array to (N, features). No effect if already flat.
    Args:
        X (np.ndarray): Array of shape (N, H, W) or (N, features)
    Returns:
        X_flat (np.ndarray): (N, H*W) or (N, features) array
    """
    n_samples = X.shape[0]
    return X.reshape(n_samples, -1)


def apply_pca(X_train_flat, X_val_flat, X_test_flat, n_components):
    """
    Fits PCA on the training data and transforms all splits to same reduced dimension.
    Args:
        X_train_flat (np.ndarray): Training data (N_train, features)
        X_val_flat (np.ndarray): Validation data (N_val, features)
        X_test_flat (np.ndarray): Test data (N_test, features)
        n_components (int): Number of PCA components to retain
    Returns:
        X_train_pca, X_val_pca, X_test_pca, pca (fitted PCA object)
    """
    pca = PCA(n_components=n_components, random_state=42)
    X_train_pca = pca.fit_transform(X_train_flat)
    X_val_pca = pca.transform(X_val_flat)
    X_test_pca = pca.transform(X_test_flat)
    return X_train_pca, X_val_pca, X_test_pca, pca

def update_config(config_path, **kwargs):
    """
    Update/add key-value pairs in an existing config file (JSON), or create if missing.
    Args:
        config_path (str): Path to config file
        kwargs: key-value pairs to write or update, e.g. processed_data_dir, pca_output_dir
    """
    # Read old config if it exists
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
    else:
        config = {}
    # Update with any new keys
    config.update(kwargs)
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)


def main(save_dir, n_components):
    config = load_config("config.json")
    data_root = config["processed_data_dir"]
    train_dir = os.path.join(data_root, 'train')
    val_dir = os.path.join(data_root, 'val')
    test_dir = os.path.join(data_root, 'test')

    print(f"Loading processed splits from: {data_root}")
    X_train, y_train = load_processed_split(train_dir)
    X_val, y_val = load_processed_split(val_dir)
    X_test, y_test = load_processed_split(test_dir)

    print("Flattening features if needed...")
    X_train_flat = flatten_images(X_train)
    X_val_flat = flatten_images(X_val)
    X_test_flat = flatten_images(X_test)

    print(f"Applying PCA with n_components={n_components} ...")
    X_train_pca, X_val_pca, X_test_pca, pca = apply_pca(
        X_train_flat, X_val_flat, X_test_flat, n_components)

    print("Saving PCA features and model...")
    # Save reduced features under chosen save_dir
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(train_dir, 'X_pca.npy'), X_train_pca)
    np.save(os.path.join(val_dir, 'X_pca.npy'), X_val_pca)
    np.save(os.path.join(test_dir, 'X_pca.npy'), X_test_pca)
    joblib.dump(pca, os.path.join(save_dir, f'pca_model_{n_components}.pkl'))
    update_config('config.json', pca_output_dir=save_dir)
    print("Dimensionality reduction complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply PCA to processed features.")
    parser.add_argument('--save_dir', type=str, default="/Users/jessica/Documents/GitHub/breast_cancer_detection/data/processed", help='Directory to save PCA model and features (default: same as processed data dir)')
    parser.add_argument('--n_components', type=int, default=70, help='Number of PCA components to use (default: 70)')
    args = parser.parse_args()
    main(args.save_dir, args.n_components)
