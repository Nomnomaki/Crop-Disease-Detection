import os
import cv2
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import glob
from tqdm import tqdm
from skimage.feature import graycomatrix, graycoprops

def extract_features(image_path):
    try:
        img = cv2.imread(image_path)
        if img is None:
            # Handle case where image couldn't be read
            print(f"Warning: Could not read image {image_path}. Skipping.")
            return None # Return None to indicate failure

        img_resized = cv2.resize(img, (224, 224))
        gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)

        # 1. Color Features (HSV Means)
        h_mean = np.mean(hsv[:,:,0])
        s_mean = np.mean(hsv[:,:,1])
        v_mean = np.mean(hsv[:,:,2])

        # 2. Texture Features (Sobel Gradient)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
        gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
        gradient_mean = np.mean(gradient_magnitude)

        # 3. Shape Features (Contour Area)
        edges = cv2.Canny(gray, 100, 200)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        area = sum(cv2.contourArea(cnt) for cnt in contours) if contours else 0 # Handle no contours case

        # 4. Texture Features (GLCM)
        # Ensure gray image is uint8 for GLCM
        gray_uint8 = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
        glcm = graycomatrix(gray_uint8, distances=[5], angles=[0], levels=256, symmetric=True, normed=True)
        contrast = graycoprops(glcm, 'contrast')[0, 0]
        dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
        homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
        energy = graycoprops(glcm, 'energy')[0, 0]
        correlation = graycoprops(glcm, 'correlation')[0, 0]

        return [
            h_mean, s_mean, v_mean, gradient_mean, area,
            contrast, dissimilarity, homogeneity, energy, correlation
        ]
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None # Return None on error

def load_dataset(data_dir):
    features = []
    labels = []
    feature_names = [
        'Hue Mean', 'Saturation Mean', 'Value Mean', 'Gradient Mean', 'Area',
        'GLCM Contrast', 'GLCM Dissimilarity', 'GLCM Homogeneity', 'GLCM Energy', 'GLCM Correlation'
    ]

    categories = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    print(f"\nFound {len(categories)} categories.")

    all_image_paths = []
    for category in categories:
        category_path = os.path.join(data_dir, category)
        # Look for common image extensions
        image_files = glob.glob(os.path.join(category_path, '*.jpg')) + \
                      glob.glob(os.path.join(category_path, '*.png')) + \
                      glob.glob(os.path.join(category_path, '*.jpeg'))
        all_image_paths.extend([(img_path, category) for img_path in image_files])

    print(f"Processing {len(all_image_paths)} images...")

    for img_path, category in tqdm(all_image_paths, desc="Loading Dataset"):
        feature_vector = extract_features(img_path)
        if feature_vector is not None: # Only add if feature extraction was successful
            features.append(feature_vector)
            labels.append(category)

    if not features:
        print("Error: No features were extracted. Please check the data directory and image files.")
        return np.array([]), np.array([]), []

    return np.array(features), np.array(labels), feature_names

def plot_confusion_matrix(y_true, y_pred, classes):
    # Ensure labels argument includes all unique classes from y_true and y_pred
    all_labels = sorted(list(set(y_true) | set(y_pred)))
    cm = confusion_matrix(y_true, y_pred, labels=all_labels)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=all_labels, yticklabels=all_labels)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()

def plot_feature_importance(model, feature_names):
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
        indices = np.argsort(importance)[::-1]
        
        plt.figure(figsize=(12, 7)) # Adjusted figure size
        plt.title('Feature Importance')
        plt.bar(range(len(importance)), importance[indices], align='center')
        plt.xticks(range(len(importance)), [feature_names[i] for i in indices], rotation=45, ha="right")
        plt.ylabel('Importance')
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        plt.close()
    else:
        print("Model does not have feature_importances_ attribute.")

def main():
    print("Starting enhanced crop disease detection model training...")

    # Load dataset
    print("\nLoading dataset...")
    X, y, feature_names = load_dataset('data')

    if X.size == 0:
        print("Exiting due to dataset loading issues.")
        return

    print(f"\nDataset loaded successfully!")
    print(f"Total samples: {len(X)}")
    if X.ndim == 2:
        print(f"Number of features: {X.shape[1]}")
    unique_labels, counts = np.unique(y, return_counts=True)
    print(f"Number of classes: {len(unique_labels)}")
    print("Class distribution:", dict(zip(unique_labels, counts)))


    # Split dataset
    print("\nSplitting dataset into train and test sets...")
    # Stratify ensures class proportions are maintained in splits
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")

    # --- Hyperparameter Tuning Setup ---
    print("\nSetting up hyperparameter tuning with GridSearchCV...")
    # Define the parameter grid to search
    param_grid = {
        'n_estimators': [100, 200, 300],          # Number of trees
        'max_depth': [None, 10, 20, 30],         # Max depth of trees
        'min_samples_split': [2, 5, 10],         # Min samples to split a node
        'min_samples_leaf': [1, 3, 5],           # Min samples in a leaf node
        'class_weight': ['balanced', None] # Handle class imbalance
    }

    # Initialize the base model
    rf = RandomForestClassifier(random_state=42)

    # Initialize GridSearchCV
    # cv=3 means 3-fold cross-validation. n_jobs=-1 uses all available CPU cores.
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid,
                               cv=3, n_jobs=-1, verbose=2, scoring='accuracy')

    # --- Train with GridSearchCV ---
    print("\nTraining model using GridSearchCV (this may take a while)...")
    grid_search.fit(X_train, y_train)

    print("\nGridSearchCV training complete.")
    print("Best parameters found: ", grid_search.best_params_)
    print("Best cross-validation accuracy: {:.2f}".format(grid_search.best_score_))

    # Get the best model found by GridSearchCV
    best_rf = grid_search.best_estimator_

    # --- Evaluate the Best Model ---
    print("\nMaking predictions on the test set using the best model...")
    y_pred = best_rf.predict(X_test)

    # Print classification report
    print("\nClassification Report (Test Set):")
    print(classification_report(y_test, y_pred, labels=sorted(list(unique_labels)), zero_division=0))

    # Plot confusion matrix
    print("\nGenerating confusion matrix...")
    plot_confusion_matrix(y_test, y_pred, classes=sorted(list(unique_labels)))

    # Plot feature importance
    print("\nGenerating feature importance plot...")
    plot_feature_importance(best_rf, feature_names)

    # --- Save the Best Model ---
    print("\nSaving the best model...")
    with open('crop_detection_model.pkl', 'wb') as f:
        pickle.dump(best_rf, f)

    print("\nTraining and evaluation completed successfully!")
    print("Best model saved as 'crop_detection_model.pkl'")
    print("Confusion matrix saved as 'confusion_matrix.png'")
    print("Feature importance plot saved as 'feature_importance.png'")

if __name__ == "__main__":
    main() 