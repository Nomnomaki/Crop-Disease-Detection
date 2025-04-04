# Crop Disease Detection System

This project implements a crop disease detection system using Random Forest and image processing techniques. It can detect various crop diseases from leaf images.

## Features

- Image processing for feature extraction
- Random Forest classification
- Performance metrics visualization
- Streamlit web interface for predictions
- Support for multiple crop types and diseases

## Installation

1. **Clone the repository:**

   ```bash
   git clone <your-repository-url>
   cd <repository-directory>
   ```

2. **Create and Activate a Virtual Environment (Recommended):**

   - It's highly recommended to use a Python version like 3.9, 3.10, or 3.11 for better compatibility with the dependencies.

   ```bash
   # Using python 3.11 as an example
   python3.11 -m venv crop
   # Activate (Windows PowerShell)
   .\crop\Scripts\Activate.ps1
   # Activate (Linux/macOS)
   # source crop/bin/activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   _This installs all necessary packages specified in `requirements.txt`, including `scikit-learn`, `opencv-python`, `streamlit`, `scikit-image`, `tqdm`, etc._

## Usage

### 1. Training the Model

- Ensure your dataset is correctly placed in the `data/` folder as described above.
- Run the training script from your terminal (make sure the virtual environment is active):
  ```bash
  python train_model.py
  ```
- **What happens during training:**
  - The script loads images from the `data/` directory.
  - It extracts the defined color, texture, and shape features for each image (progress shown via `tqdm`).
  - The dataset is split into training and testing sets.
  - **GridSearchCV** searches for the best Random Forest hyperparameters using cross-validation (this step can take some time, progress might be shown depending on `verbose` level).
  - The best model found by GridSearchCV is trained on the full training set.
  - The best model is evaluated on the test set.
  - The classification report is printed to the console.
  - `confusion_matrix.png` and `feature_importance.png` are generated and saved.
  - The best-trained model is saved as `crop_detection_model.pkl`.

### 2. Making Predictions with the Web App

- After successful training (the `crop_detection_model.pkl` file exists), run the Streamlit app:
  ```bash
  streamlit run app.py
  ```
- Open your web browser and navigate to the local URL provided by Streamlit (usually `http://localhost:8501`).
- Use the file uploader to select an image of a crop leaf.
- The app will:
  - Display the uploaded image.
  - Extract features from the image.
  - Load the saved model (`crop_detection_model.pkl`).
  - Predict the condition using the model.
  - Show the prediction, confidence scores (probabilities), and feature importance plots.

## Model Features

The model extracts the following features from images:

- Color features (HSV means)
- Texture features (Sobel gradient)
- Shape features (contour area)

## Performance Metrics

The model generates:

- Confusion matrix
- Classification report
- Feature importance plot

## License

This project is licensed under the MIT License - see the LICENSE file for details (if one exists).
