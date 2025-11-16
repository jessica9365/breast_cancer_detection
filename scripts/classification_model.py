import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import json
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
import joblib
import lightgbm as lgb
import xgboost as xgb

#Important class names are declared globally for code clarity 
class_names = ["Benign", "Malignant", "Normal"]

# Data Loading
def load_pca_data(split_dir):
    X = np.load(os.path.join(split_dir, 'X_pca.npy'))
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

def evaluate_model(clf, X, y, set_name="Validation"):
    y_pred = clf.predict(X)
    y_prob = clf.predict_proba(X)[:, 1] if len(clf.classes_) == 2 else clf.predict_proba(X)
    acc = accuracy_score(y, y_pred)
    print(f"{set_name} Accuracy: {acc:.3f}")
    print(f"{set_name} Classification Report:\n", classification_report(y, y_pred))
    cm = confusion_matrix(y, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{set_name} Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
    return y_pred, y_prob

def plot_roc_curve(y, y_prob, set_name="Validation"):
    if len(np.unique(y)) == 2:
        fpr, tpr, thresholds = roc_curve(y, y_prob)
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Chance')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'{set_name} ROC Curve')
        plt.legend(loc="lower right")
        plt.show()
        print(f"{set_name} ROC AUC: {roc_auc:.3f}")

def plot_precision_recall(y_true, y_scores, class_names=None):
    """
    Plots precision-recall curves for each class.
    Args:
        y_true (array-like): True labels (N,)
        y_scores (array-like): Shape (N, n_classes) predicted probabilities
        class_names (list): List of class names for legend
    """
    n_classes = y_scores.shape[1]
    plt.figure()
    for i in range(n_classes):
        precision, recall, _ = precision_recall_curve((y_true == i).astype(int), y_scores[:, i])
        ap = average_precision_score((y_true == i).astype(int), y_scores[:, i])
        label = f"Class {i}" if class_names is None else class_names[i]
        plt.plot(recall, precision, lw=2, label=f'{label} (AP={ap:.2f})')
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall curve")
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()

def train_logistic_regression(X_train, y_train):
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train, y_train)
    return clf

def train_knn_weighted(X_train, y_train, k=5):
    # Use weights="distance" so nearer neighbors count more
    knn = KNeighborsClassifier(n_neighbors=k, weights='distance')
    knn.fit(X_train, y_train)
    return knn

def train_svm(X_train, y_train, kernel='rbf', C=1.0, gamma='scale'):
    """
    Trains SVM with specified kernel and hyperparameters.
    multiclass is handled by one-vs-one automatically by scikit-learn's SVC.
    """
    svm = SVC(kernel=kernel, C=C, gamma=gamma, probability=True, class_weight='balanced', random_state=42)
    svm.fit(X_train, y_train)
    return svm

def train_lightGBM(X_train, y_train, X_val, y_val):
    lgb_clf = lgb.LGBMClassifier(objective='multiclass', num_class=3, random_state=42, n_estimators=100)
    lgb_clf.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='multi_logloss',
        callbacks=[lgb.early_stopping(20)],
    )

    return lgb_clf

def train_XGBoost(X_train, y_train, X_val, y_val):
    # Initialize model
    xgb_clf = xgb.XGBClassifier(
        objective='multi:softprob',
        num_class=3,
        random_state=42,
        n_estimators=100,
        eval_metric='mlogloss',
        early_stopping_rounds=20
    )

    xgb_clf.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=True
    )

    return xgb_clf

def update_results_config(json_path, results_dir, model_paths):
        # Read (or create) and update results info
        config = {}
        if os.path.exists(json_path):
            with open(json_path, "r") as f:
                config = json.load(f)
        config["results_dir"] = results_dir
        config["model_paths"] = model_paths
        with open(json_path, "w") as f:
            json.dump(config, f, indent=2)

def save_results_text(filename, y_true, y_pred, model_name):
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=class_names)
    cm = confusion_matrix(y_true, y_pred)
    with open(filename, "w") as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Accuracy: {acc:.3f}\n\n")
        f.write("Classification Report:\n")
        f.write(report + "\n")
        f.write("Confusion Matrix:\n")
        np.savetxt(f, cm, fmt='%d')

def main():
    parser = argparse.ArgumentParser(description="Run and evaluate classification models on PCA-reduced features.")
    parser.add_argument('--results_dir', type=str, default='/Users/jessica/Documents/GitHub/breast_cancer_detection/results', help='Folder to save results and models')
    parser.add_argument('--k', type=int, default=3, help='Best k for kNN (default: 3)')
    parser.add_argument('--svm_kernel', type=str, default='rbf', help='Kernel type for SVM (default: rbf)')
    parser.add_argument('--svm_C', type=float, default=1.0, help='Penalty parameter C for SVM (default: 1.0)')
    parser.add_argument('--svm_gamma', type=str, default='scale', help='Gamma for SVM (default: scale or auto)')
    parser.add_argument('--config_path', type=str, default='config.json', help='Path to pipeline config file (default: ../config.json)')
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    config = load_config(args.config_path)
    data_root = config["pca_output_dir"]
    train_dir = os.path.join(data_root, 'train')
    val_dir = os.path.join(data_root, 'val')
    test_dir = os.path.join(data_root, 'test')
    X_train, y_train = load_pca_data(train_dir)
    X_val, y_val = load_pca_data(val_dir)
    X_test, y_test = load_pca_data(test_dir)

    # --- Logistic Regression ---
    logreg = train_logistic_regression(X_train, y_train)
    y_pred_lr, y_prob_lr = evaluate_model(logreg, X_test, y_test, set_name="LogReg Test")
    logreg_dir = os.path.join(args.results_dir, "Logistic_Regression")
    os.makedirs(logreg_dir, exist_ok=True)
    model_path_lr = os.path.join(logreg_dir, "logreg_model.pkl")
    joblib.dump(logreg, model_path_lr)

    # --- kNN ---
    knn = train_knn_weighted(X_train, y_train, k=args.k)
    y_pred_knn, y_prob_knn = evaluate_model(knn, X_test, y_test, set_name="kNN Test")
    knn_dir = os.path.join(args.results_dir, "knn")
    os.makedirs(knn_dir, exist_ok=True)
    model_path_knn = os.path.join(knn_dir, "knn_model.pkl")
    joblib.dump(knn, model_path_knn)

    # --- SVM ---
    svm = train_svm(X_train, y_train, kernel=args.svm_kernel, C=args.svm_C, gamma=args.svm_gamma)
    y_pred_svm, y_prob_svm = evaluate_model(svm, X_test, y_test, set_name="SVM Test")
    svm_dir = os.path.join(args.results_dir, "svm")
    os.makedirs(svm_dir, exist_ok=True)
    model_path_svm = os.path.join(svm_dir, "svm_model.pkl")
    joblib.dump(svm, model_path_svm)

    # --- LightGBM ---
    lgb_clf = train_lightGBM(X_train, y_train, X_val, y_val)
    y_pred_lightGBM, y_prob_lightGBM = evaluate_model(lgb_clf, X_test, y_test, set_name="LightGBM Test")
    lgb_dir = os.path.join(args.results_dir, "LightGBM")
    os.makedirs(lgb_dir, exist_ok=True)
    model_path_lgb = os.path.join(lgb_dir, "LightGBM_model.pkl")
    joblib.dump(lgb_clf, model_path_lgb)

    # --- XGBoost ---
    xgb_clf = train_XGBoost(X_train, y_train, X_val, y_val)
    y_pred_XGBoost, y_prob_XGBoost = evaluate_model(xgb_clf, X_test, y_test, set_name="XGBoost Test")
    xgb_dir = os.path.join(args.results_dir, "XGBoost")
    os.makedirs(xgb_dir, exist_ok=True)
    model_path_xgb = os.path.join(xgb_dir, "XGBoost_model.pkl")
    joblib.dump(xgb_clf, model_path_xgb)

    save_results_text(os.path.join(logreg_dir, "logreg_results.txt"), y_test, y_pred_lr, "LogisticRegression")
    save_results_text(os.path.join(knn_dir, "knn_results.txt"), y_test, y_pred_knn, f"kNN_k={args.k}")
    save_results_text(os.path.join(svm_dir, "svm_results.txt"), y_test, y_pred_svm, f"SVM_{args.svm_kernel}")
    save_results_text(os.path.join(lgb_dir, "lightGBM_results.txt"), y_test, y_pred_lightGBM, "LightGBM")
    save_results_text(os.path.join(xgb_dir, "XGBoost_results.txt"), y_test, y_pred_XGBoost, "XGBoost")

    update_results_config(os.path.join(args.results_dir, "results_config.json"),
                         args.results_dir,
                         {"logistic": model_path_lr, "knn": model_path_knn, "svm": model_path_svm, "lightGBM": model_path_lgb , "XGBoost": model_path_xgb})
    
    print(f"All results and models have been saved in: {args.results_dir}")
    
if __name__ == "__main__":
    main()