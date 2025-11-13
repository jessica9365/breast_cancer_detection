import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import re
from imgaug import augmenters as iaa
from sklearn.model_selection import train_test_split

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
    
def concatenate_mask_channel(image, mask):
    """
    Adds mask as a second channel to the image.
    Args:
        image (np.array): Grayscale image.
        mask (np.array): Grayscale mask or None.
    Returns:
        img_with_mask (np.array): 2-channel image (image, mask).
    """
    if mask is None:
        mask = np.zeros_like(image)
    return np.stack([image, mask], axis=-1)

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
        X (np.array): Image data array (N, H, W) or (N, H, W, C).
        y (np.array): Labels array (N,).
        save_dir (str): Directory to save .npy files.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    np.save(os.path.join(save_dir, "X.npy"), X)
    np.save(os.path.join(save_dir, "y.npy"), y)


def main():
    # Specify dataset root and classes
    data_dir = "/Users/jessica/Documents/GitHub/breast_cancer_detection/data/raw"  # adjust path
    classes = ["Benign", "Malignant", "Normal"]  # adjust based on your folder names

    # Load images, masks, and labels
    images, masks, labels = load_images_with_masks(data_dir, classes)

    processed_imgs = []

    for img, mask in zip(images, masks):
        # Crop image by lesion mask
        cropped = apply_mask_and_crop(img, mask)

        # # Concatenate mask as channel
        # img_masked = concatenate_mask_channel(cropped, mask)

        # Resize to consistent shape
        resized = resize_image(cropped, target_size=(128, 128))

        # Normalize pixel values
        normed = normalize_image(resized)

        # Optionally augment data here, e.g., only on training set
        aug = augment_image(normed)  # Uncomment to apply augmentation

        processed_imgs.append(aug)  # Or aug

    # Convert lists to NumPy arrays
    X = np.array(processed_imgs)
    y = np.array(labels)

    # Split data: 70% train, 15% validation, 15% test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y)

    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.1765, random_state=42, stratify=y_trainval)  # 0.1765 ~15/85

    # Save datasets separately
    save_dir = "/Users/jessica/Documents/GitHub/breast_cancer_detection/data/processed" # adjust path
    save_processed_data(X_train, y_train, os.path.join(save_dir, "train"))
    save_processed_data(X_val, y_val, os.path.join(save_dir, "val"))
    save_processed_data(X_test, y_test, os.path.join(save_dir, "test"))

    print("Preprocessing complete. Data saved in train, val, test folders.")


if __name__ == '__main__':
    main()
