import os
import cv2
import numpy as np
import argparse
from imgaug import augmenters as iaa
from sklearn.model_selection import train_test_split
from skimage.feature import graycomatrix, graycoprops
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import json
import re


def load_images_with_masks(data_dir, classes):
    """
    Loads images and corresponding mask files from class folders, assigns integer labels.
    Args:
        data_dir (str): Path to the data directory with class subfolders.
        classes (list): List of class names (folder names).
    Returns:
        images (list): List of loaded images.
        masks (list): List of loaded mask images (or None for normal).
        labels (list): Integer label for each sample.
    """
    images, masks, labels = [], [], []
    for label, cls in enumerate(classes):
        folder = os.path.join(data_dir, cls)
        for fname in os.listdir(folder):
            if fname.endswith('.png') and '_mask' not in fname:
                parts = fname.split('.')[0].split('_')
                if len(parts) != 2:
                    print(f"Skipping file with unexpected format: {fname}")
                    continue
                num = parts[1]
                mask_name = f"{cls}_{num}_mask.png"
                mask_path = os.path.join(folder, mask_name) if os.path.exists(os.path.join(folder, mask_name)) else None
                img = cv2.imread(os.path.join(folder, fname), cv2.IMREAD_GRAYSCALE)
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) if mask_path else None
                images.append(img)
                masks.append(mask)
                labels.append(label)
    return images, masks, labels

def rename_files_for_consistency(folder, class_name):
    for fname in os.listdir(folder):
        if fname.endswith('.png'):
            # Look for patterns like benign (12).png or benign (12)_mask.png
            match = re.match(rf'{class_name} \((\d+)\)(.*?)\.png', fname)
            if match:
                num = match.group(1)
                suffix = match.group(2)  # '_mask' or ''
                # Build new filename
                new_fname = f'{class_name}_{num}{suffix}.png'
                os.rename(os.path.join(folder, fname), os.path.join(folder, new_fname))
                # print(f'Renamed: {fname} -> {new_fname}')


def apply_mask_and_crop(image, mask):
    """
    Crops the input image to the bounding box of the mask region.
    Args:
        image (np.array): Grayscale image array.
        mask (np.array): Corresponding mask array (white lesion, black background).
    Returns:
        cropped_img (np.array): Cropped image array.
    """
    if mask is None:
        return image
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        x, y, w, h = cv2.boundingRect(contours[0])
        cropped_img = image[y:y+h, x:x+w]
        return cropped_img
    else:
        return image  # fallback if no contour found
    
def resize_image(image, target_size=(128, 128)):
    """
    Resizes image to target shape for consistent model input.
    Args:
        image (np.array): Image array.
        target_size (tuple): Desired output size (height, width).
    Returns:
        resized_img (np.array): Resized image array.
    """
    return cv2.resize(image, target_size)

def normalize_image(image):
    """
    Normalizes pixel intensity values to [0, 1] float.
    Args:
        image (np.array): Image array, usually uint8.
    Returns:
        norm_img (np.array): Image with pixel values in [0, 1].
    """
    return image.astype(np.float32) / 255.0

def extract_glcm_features(image):
    """
    Extracts basic texture features using the Gray-Level Co-occurrence Matrix (GLCM).
    Args:
        image (np.array): Input grayscale image, normalized in [0, 1] or uint8.
    Returns:
        features (np.array): Array of 4 scalar texture features: contrast, homogeneity, energy, correlation (shape: (4,)).
    Notes:
        - Converts input to uint8 if necessary for GLCM calculation.
        - Uses a distance of 1 and angle of 0 degrees for GLCM.
    """
    if image.ndim == 3:
        image = image[..., 0]
    glcm = graycomatrix((image*255).astype('uint8'),
                       distances=[1], angles=[0], levels=256,
                       symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0,0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0,0]
    energy = graycoprops(glcm, 'energy')[0,0]
    correlation = graycoprops(glcm, 'correlation')[0,0]
    return np.array([contrast, homogeneity, energy, correlation], dtype=np.float32) 

def augment_image(image):
    """
    Applies random augmentation (flip, rotate, affine).
    Args:
        image (np.array): Image array, typically normalized.
    Returns:
        aug_img (np.array): Augmented image.
    """
    seq = iaa.Sequential([
        iaa.Fliplr(0.5),             # Horizontal flip 50%
        iaa.Rotate((-10, 10)),       # Random rotation
        iaa.Affine(
            scale=(0.9, 1.1),        # Random scaling
            translate_percent=(-0.05, 0.05)
        )
    ])
    return seq.augment_image(image)

def save_processed_data(X, y, save_dir):
    """
    Saves processed images and labels to disk as NumPy arrays.
    Args:
        X (np.array): Image/feature data array (N, features).
        y (np.array): Labels array (N,).
        save_dir (str): Directory to save .npy files.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    np.save(os.path.join(save_dir, "X.npy"), X)
    np.save(os.path.join(save_dir, "y.npy"), y)

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


def main(data_dir, save_dir):
    # Class names (adjust as needed)
    classes = ["Benign", "Malignant", "Normal"]

    # Rename files for consistency
    data_dir = 'data_dir'
    for cls in ['benign', 'malignant', 'normal']:
        folder = os.path.join(data_dir, cls)
        rename_files_for_consistency(folder, cls)

    # Load images, masks, and labels
    images, masks, labels = load_images_with_masks(data_dir, classes)

    all_features = []
    for img, mask in zip(images, masks):
        cropped = apply_mask_and_crop(img, mask)
        resized = resize_image(cropped, target_size=(128, 128))
        normed = normalize_image(resized)
        # Optionally augment: aug = augment_image(normed)

        features = extract_glcm_features(normed)
        features_scaled = (features - np.mean(features)) / (np.std(features) + 1e-8)
        all_features.append(np.concatenate([normed.flatten(), features_scaled]))

    X = np.array(all_features)
    y = np.array(labels)

    # Oversample minority then undersample majority for balance
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    rus = RandomUnderSampler(random_state=42)
    X_balanced, y_balanced = rus.fit_resample(X_resampled, y_resampled)

    # Split data: 70% train, 15% val, 15% test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X_balanced, y_balanced, test_size=0.15, random_state=42, stratify=y_balanced)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.1765, random_state=42, stratify=y_trainval)

    # Save datasets
    save_processed_data(X_train, y_train, os.path.join(save_dir, "train"))
    save_processed_data(X_val, y_val, os.path.join(save_dir, "val"))
    save_processed_data(X_test, y_test, os.path.join(save_dir, "test"))
    update_config('config.json', processed_data_dir=save_dir)
    print("Preprocessing complete. Data saved in train, val, test folders.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Preprocess breast ultrasound dataset.")
    parser.add_argument('--data_dir', type=str, default="/Users/jessica/Documents/GitHub/breast_cancer_detection/data/raw", help='Root data folder with class subfolders')
    parser.add_argument('--save_dir', type=str, default="/Users/jessica/Documents/GitHub/breast_cancer_detection/data/processed", help='Folder to save processed splits')
    args = parser.parse_args()
    main(args.data_dir, args.save_dir)