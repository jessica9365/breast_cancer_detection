# Breast Cancer Ultrasound Detection

A modular pipeline for preprocessing, dimensionality reduction, and classification of high-dimensional medical ultrasound images for breast cancer diagnosis.

## Overview

This project provides a reproducible ML workflow to distinguish Benign, Malignant, and Normal breast ultrasound cases using classical machine learning models. Scripts and notebooks are organized for clarity, experiment tracking, and flexible configuration.

##### Key Features

1. Flexible, command-line controlled preprocessing (preprocess.py) with GLCM texture features and class balancing
2. Dimensionality reduction (dimensionality_reduction.py) using PCA (configurable number of components)
3. Multiple classifiers: Logistic Regression, k-Nearest Neighbor (kNN), Support Vector Machine (SVM)
4. Clean separation of exploratory .ipynb notebooks and reproducible .py scripts
5. All paths and parameters configurable via CLI and automatically tracked in config/results files
6. Visualizations for model performance (confusion matrix, precision-recall, ROC; in notebooks)

## Getting Started

1. Clone/download this repository
2. Download the dataset: Al-Dhabyani W, Gomaa M, Khaled H, Fahmy A. Dataset of breast ultrasound images. Data in Brief. 2020 Feb;28:104863. DOI: 10.1016/j.dib.2019.104863. Kaggle Link: https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset
3. Place the data folders under: text/data/raw/
4. Set up the Python environment using provided requirements.txt
5. Run preprocessing to save processed data and update config.json for pipeline tracking.

```python
python preprocess.py --data_dir data/raw --save_dir data/processed
```

6. Run dimensionality reduction for PCA-reduced features and updating config file.

```python
python dimensionality_reduction.py --save_dir data/processed --n_components 70
```

7. Run classification_model.py to train and evaluate models

```python
python classification_model.py --results_dir results --k 1 --svm_kernel linear --svm_C 1.0 --svm_gamma scale
```

8. Use Jupyter notebooks (notebooks/) for visualization (choosing components, viewing confusion matrices, ROC/PR curves, etc).

## Directory Structure

project_root/
├── preprocess.py # Preprocessing pipeline
├── dimensionality_reduction.py # PCA pipeline
├── classification_model.py # Model training & evaluation
├── requirements.txt # Project dependencies
├── notebooks/ # Exploratory analysis notebooks
├── config.json # For pipeline path tracking
├── results/ # Models, result files, results config
└── data/
├── raw/ # Original images/masks
├── processed/ # Preprocessed and split features

## Reproducibility and Tracking

1. All scripts read/write config.json and update with output paths
2. Results and trained models are saved with clear filenames

## Customisation and Tracking

1. All major parameters (paths, model hyperparameters, PCA components) are configurable via command line
2. Easily extend with more radiomics features, other classifiers, or dimensionality reduction methods
3. The pipeline is robust to changing paths and supports collaborative/iterative workflows

## Acknowledgements

1. Dataset: Al-Dhabyani W, Gomaa M, Khaled H, Fahmy A. Dataset of breast ultrasound images. Data in Brief. 2020 Feb;28:104863. DOI: 10.1016/j.dib.2019.104863. Kaggle: https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset
2. Built using open-source Python scientific libraries
