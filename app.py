import os
from flask import Flask, request, render_template, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import io # NEW: Import the io module

# --- Configuration Paths ---
# These paths are relative to where app.py is located, which is /content/SmartSortingProject
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# The exact name of the folder inside 'dataset' that contains your class subdirectories
DATASET_BASE_PATH = 'Fruit And Vegetable Diseases Dataset' # This is based on previous outputs

# This is the full path to the directory containing your image class folders (e.g., Apple_Healthy, Banana_Rotten)
# This is important for correctly determining the class names.
IMAGES_CONTAINING_ROOT = os.path.join(PROJECT_ROOT, 'dataset', DATASET_BASE_PATH)

# Path to your saved model file in Google Drive
# IMPORTANT: Ensure this path is correct and your Drive is mounted!
MODEL_PATH = '/content/drive/MyDrive/SmartSortingProject_Colab/healthy_vs_rotten.h5' # This path is consistent with your training output

# Image dimensions your model expects (from train_model.py)
IMG_HEIGHT = 224
IMG_WIDTH = 224

# --- Load the trained model ---
model = None # Initialize model to None
try:
    model = load_model(MODEL_PATH)
    print(f"Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Please ensure the model file exists at the specified path and is not corrupted.")
    # If the model cannot be loaded, the app cannot function correctly.

# --- Define your class names ---
# This is CRUCIAL. The order MUST match the class indexing used during training.
# ImageDataGenerator.flow_from_directory typically sorts alphabetically by folder name.
# So, dynamically fetching and sorting them from your dataset folder is the safest way.
class_names = []
if os.path.exists(IMAGES_CONTAINING_ROOT):
    class_names = sorted([d for d in os.listdir(IMAGES_CONTAINING_ROOT) if os.path.isdir(os.path.join(IMAGES_CONTAINING_ROOT, d))])
    print(f"Discovered class names: {class_names}")
    if not class_names:
        print("WARNING: No class directories found in IMAGES_CONTAINING_ROOT. Check dataset structure.")
else:
    print(f"ERROR: IMAGES_CONTAINING_ROOT '{IMAGES_CONTAINING_ROOT}' does not exist. Cannot determine class names.")
    # This block provides a fallback if the dataset path isn't found,
    # but based on your previous checks, your dataset path appears correct.

# --- Flask App Initialization ---
# `template_folder` and `static_folder` point to subdirectories relative to app.py
app = Flask(__name__, template_folder='templates', static_folder='static')

# --- Routes ---
@app.route('/')
def index():
    """Renders the main HTML page for image upload."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handles image upload, runs prediction, and returns results as JSON."""
    if model is None:
        return jsonify({"error": "Model not loaded. Please check server logs."}), 500

    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request."}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file."}), 400
    
    if file:
        try:
            # NEW: Read the file content into a BytesIO object
            # This handles the <class 'tempfile.SpooledTemporaryFile'> issue
            img_stream = io.BytesIO(file.read()) 
            
            # Load the image and resize it to the target size from the BytesIO stream
            img = image.load_img(img_stream, target_size=(IMG_HEIGHT, IMG_WIDTH))
            
            # Convert the image to a NumPy array
            img_array = image.img_to_array(img)
            
            # Expand dimensions to create a batch (model expects a batch of images)
            img_array = np.expand_dims(img_array, axis=0) 
            
            # Normalize the image pixel values from [0, 255] to [0, 1]
            # This MUST match the preprocessing during model training (rescale=1./255)
            img_array /= 255.0 

            # Make prediction using the loaded model
            predictions = model.predict(img_array)
            
            # Get the index of the class with the highest probability
            predicted_class_index = np.argmax(predictions[0])
            
            # Map the index to the human-readable class name
            predicted_class_name = class_names[predicted_class_index]
            
            # Calculate confidence percentage
            confidence = np.max(predictions[0]) * 100

            # Return the prediction result as a JSON response
            return jsonify({
                "predicted_class": predicted_class_name,
                "confidence": f"{confidence:.2f}%"
            })
        except Exception as e:
            # Catch any errors during image processing or prediction
            return jsonify({"error": f"Error processing image or prediction: {e}"}), 500
    
    return jsonify({"error": "An unexpected error occurred."}), 500


# --- Main execution block for running the Flask app ---
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)