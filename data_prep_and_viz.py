import os
import zipfile
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# --- Configuration (MUST BE UPDATED BY USER) ---
# The exact name of your downloaded Kaggle ZIP file
ZIP_FILE_NAME = 'archive.zip' # This should match your zip file name

# The top-level folder name inside the zip after extraction.
# Based on your input, this is 'Fruit and Vegetable Diseases Dataset'
DATASET_BASE_PATH = 'Fruit and Vegetable Diseases Dataset'

# Base directory for the project
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Paths for dataset
DATASET_ZIP_PATH = os.path.join(PROJECT_ROOT, 'dataset', ZIP_FILE_NAME)
DATASET_EXTRACT_PATH = os.path.join(PROJECT_ROOT, 'dataset')

# IMPORTANT CHANGE: TRAIN_DIR and TEST_DIR now point directly to the base dataset folder,
# as it contains the class subdirectories directly, without 'train'/'test' subfolders.
TRAIN_DIR = os.path.join(DATASET_EXTRACT_PATH, DATASET_BASE_PATH)
TEST_DIR = os.path.join(DATASET_EXTRACT_PATH, DATASET_BASE_PATH) # Test will use validation_split from training data

# Image parameters
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.2 # 20% of the data will be used for validation

# --- 1. Unzip the Dataset ---
print(f"Checking for and unzipping dataset from {DATASET_ZIP_PATH}...")
if os.path.exists(DATASET_ZIP_PATH):
    if not os.path.exists(os.path.join(DATASET_EXTRACT_PATH, DATASET_BASE_PATH)):
        print("Dataset not found extracted. Unzipping now...")
        with zipfile.ZipFile(DATASET_ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(DATASET_EXTRACT_PATH)
        print("Dataset unzipped successfully!")
    else:
        print(f"Dataset folder '{DATASET_BASE_PATH}' already exists in '{DATASET_EXTRACT_PATH}'. Skipping unzip.")
else:
    print(f"Error: ZIP file not found at {DATASET_ZIP_PATH}. Please ensure you have downloaded the dataset and placed it in the 'dataset' folder.")
    exit()

# --- 2. Setup Data Generators ---
print("\nSetting up data generators...")

# Data augmentation for training data to prevent overfitting
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=VALIDATION_SPLIT # Split data for training and validation
)

# Only rescaling for test/validation data - no augmentation
test_datagen = ImageDataGenerator(rescale=1./255) # Test_datagen for general use if needed, but validation comes from train_datagen

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR, # Now points directly to the dataset base folder
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training', # Specify this is the training subset
    shuffle=True
)

validation_generator = train_datagen.flow_from_directory(
    TRAIN_DIR, # Points to the same base folder
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation', # Specify this is the validation subset
    shuffle=False # No need to shuffle validation data
)

# This test_generator will technically load the full dataset base,
# but it's used for visualization or if you later want to predict
# on the full un-split data. For model evaluation, use validation_generator.
test_generator = test_datagen.flow_from_directory(
    TEST_DIR, # Points directly to the dataset base folder
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False # No need to shuffle test data for evaluation
)


print(f"\nDetected {train_generator.num_classes} classes.")
print("Class indices:", train_generator.class_indices)

# Save class names for app.py and train_model.py
CLASS_NAMES = sorted(train_generator.class_indices.keys())
print("\n------------------------------------------------------------------")
print("IMPORTANT: Copy the 'CLASS_NAMES' list below EXACTLY for app.py and train_model.py:")
print(CLASS_NAMES)
print("------------------------------------------------------------------")

# --- 3. Visualize Sample Images ---
print("\nVisualizing sample images...")

def plot_images(images, labels, class_names):
    plt.figure(figsize=(10, 10))
    for i in range(min(9, len(images))): # Plot up to 9 images
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i])
        # Get the actual class name from the one-hot encoded label
        predicted_class_index = np.argmax(labels[i])
        class_name = class_names[predicted_class_index]
        plt.title(class_name)
        plt.axis("off")
    plt.show()

# Get a batch of training images and labels
sample_images, sample_labels = next(train_generator)

# Plot the images
plot_images(sample_images, sample_labels, CLASS_NAMES)

print("\nData pre-processing and visualization script finished.")