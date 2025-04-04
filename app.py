import streamlit as st
import cv2
import numpy as np
import pickle
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
from skimage.feature import graycomatrix, graycoprops

# Function to extract features from image
def extract_features(image):
    try:
        # Check if image is valid
        if image is None:
            st.error("Error: Could not process image.")
            return None

        # Resize image
        img_resized = cv2.resize(image, (224, 224))
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
        area = sum(cv2.contourArea(cnt) for cnt in contours) if contours else 0

        # 4. Texture Features (GLCM)
        gray_uint8 = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
        glcm = graycomatrix(gray_uint8, distances=[5], angles=[0], levels=256, symmetric=True, normed=True)
        contrast = graycoprops(glcm, 'contrast')[0, 0]
        dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
        homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
        energy = graycoprops(glcm, 'energy')[0, 0]
        correlation = graycoprops(glcm, 'correlation')[0, 0]

        return np.array([[
            h_mean, s_mean, v_mean, gradient_mean, area,
            contrast, dissimilarity, homogeneity, energy, correlation
        ]])
    except Exception as e:
        st.error(f"Feature extraction failed: {e}")
        return None

# Load the trained model
@st.cache_resource
def load_model(model_path='crop_detection_model.pkl'):
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        st.error(f"Error: Model file '{model_path}' not found. Please train the model first.")
        return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def plot_feature_importance_app(model, feature_names):
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
        indices = np.argsort(importance)[::-1]
        
        # Create DataFrame for plotting
        importance_df = pd.DataFrame({
            'Feature': [feature_names[i] for i in indices],
            'Importance': importance[indices]
        }).set_index('Feature')

        st.bar_chart(importance_df)
    else:
        st.warning("Model does not have feature importance information.")

def main():
    # --- Page Configuration ---
    st.set_page_config(page_title="Crop Disease Detection", layout="wide")

    # --- Sidebar ---
    st.sidebar.title("About")
    st.sidebar.info(
        """
        **Crop Disease Detection App**

        This app uses a Random Forest model trained on image features
        (color, texture, shape) to predict potential diseases in crop leaves.

        **Features Used:**
        - HSV Color Means
        - Sobel Gradient Mean
        - Contour Area
        - GLCM Texture Properties

        Upload an image of a crop leaf to get a prediction.
        """
    )
    st.sidebar.markdown("---") # Divider

    # --- Main Page Title ---
    st.title("🌿 Crop Disease Detection")
    st.markdown("Upload an image of a crop leaf to analyze its condition.")

    # --- File Uploader ---
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # --- Load Model ---
        model = load_model()
        if model is None:
            st.stop() # Stop execution if model failed to load

        # --- Display Image and Results in Columns ---
        col1, col2 = st.columns([1, 1]) # Adjust ratio as needed

        with col1:
            st.subheader("Uploaded Image")
            try:
                image = Image.open(uploaded_file)
                # Convert PIL image to OpenCV format (RGB -> BGR)
                image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                st.image(image, caption='Uploaded Image', use_column_width=True)
            except Exception as e:
                st.error(f"Error displaying image: {e}")
                st.stop()

        with col2:
            st.subheader("Analysis Results")
            with st.spinner('Analyzing image and making prediction...'):
                # Extract features
                features = extract_features(image_cv)

                if features is not None:
                    # Make prediction
                    try:
                        prediction = model.predict(features)[0]
                        probabilities = model.predict_proba(features)[0]
                        classes = model.classes_

                        st.success(f"**Predicted Condition:** {prediction}")

                        # Display Probabilities as a Bar Chart
                        st.markdown("---") # Divider
                        st.markdown("**Prediction Confidence:**")
                        prob_df = pd.DataFrame({
                            'Class': classes,
                            'Probability': probabilities
                        }).set_index('Class')
                        st.bar_chart(prob_df)

                        # Display Feature Importance
                        st.markdown("---") # Divider
                        st.markdown("**Feature Importance for Prediction:**")
                        # Define feature names (must match training)
                        feature_names = [
                            'Hue Mean', 'Saturation Mean', 'Value Mean', 'Gradient Mean', 'Area',
                            'GLCM Contrast', 'GLCM Dissimilarity', 'GLCM Homogeneity', 'GLCM Energy', 'GLCM Correlation'
                        ]
                        plot_feature_importance_app(model, feature_names)

                    except Exception as e:
                        st.error(f"Prediction failed: {e}")
                else:
                    # Error message already shown by extract_features
                    st.warning("Could not proceed with prediction due to feature extraction error.")

    else:
        st.info("Please upload an image file to start the analysis.")

if __name__ == "__main__":
    main() 