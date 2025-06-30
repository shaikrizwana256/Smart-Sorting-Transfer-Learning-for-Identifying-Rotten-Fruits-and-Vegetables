import os
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt

# --- Configuration (MUST MATCH data_prep_and_viz.py) ---
# The top-level folder name inside the zip after extraction.
# This MUST be exactly the same as in data_prep_and_viz.py
DATASET_BASE_PATH = 'Fruit and Vegetable Diseases Dataset'

# Base directory for the project
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Paths for dataset (MUST MATCH data_prep_and_viz.py adjustments)
DATASET_EXTRACT_PATH = os.path.join(PROJECT_ROOT, 'dataset')
TRAIN_DIR = os.path.join(DATASET_EXTRACT_PATH, DATASET_BASE_PATH) # Pointing to the base dataset folder
TEST_DIR = os.path.join(DATASET_EXTRACT_PATH, DATASET_BASE_PATH)   # Pointing to the base dataset folder

# Image parameters (MUST MATCH data_prep_and_viz.py)
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.2 # Must match data_prep_and_viz.py

# Model parameters
NUM_EPOCHS = 20 # You can adjust this based on training progress
LEARNING_RATE = 0.0001
MODEL_SAVE_PATH = os.path.join(PROJECT_ROOT, 'healthy_vs_rotten.h5')

# IMPORTANT: This list MUST be exactly copied from the output of data_prep_and_viz.py
# Example (REPLACE WITH YOUR ACTUAL OUTPUT):
CLASS_NAMES = ['Apple_Healthy', 'Apple_Rotten', 'Banana_Healthy', 'Banana_Rotten',
               'Bellpepper_Healthy', 'Bellpepper_Rotten', 'Carrot_Healthy', 'Carrot_Rotten',
               'Cucumber_Healthy', 'Cucumber_Rotten', 'Grape_Healthy', 'Grape_Rotten',
               'Guava_Healthy', 'Guava_Rotten', 'Jujube_Healthy', 'Jujube_Rotten',
               'Mango_Healthy', 'Mango_Rotten', 'Orange_Healthy', 'Orange_Rotten',
               'Pomegranate_Healthy', 'Pomegranate_Rotten', 'Potato_Healthy', 'Potato_Rotten',
               'Strawberry_Healthy', 'Strawberry_Rotten', 'Tomato_Healthy', 'Tomato_Rotten']
NUM_CLASSES = len(CLASS_NAMES)


# --- 1. Load Data Generators ---
print("Loading data generators...")
# Re-define ImageDataGenerators, ensuring validation_split matches
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=VALIDATION_SPLIT # Use the same split
)

# Test/Validation data only needs rescaling
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

validation_generator = train_datagen.flow_from_directory(
    TRAIN_DIR, # Validation comes from the same base directory
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

# For final evaluation/prediction on the "test" data (which is the full dataset in your case)
# This generator will be used for final model evaluation or when predicting new images
full_data_generator = test_datagen.flow_from_directory(
    TEST_DIR, # Points to the full dataset base
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

print(f"Found {train_generator.samples} training images belonging to {train_generator.num_classes} classes.")
print(f"Found {validation_generator.samples} validation images belonging to {validation_generator.num_classes} classes.")


# --- 2. Build the VGG16 Model (Transfer Learning) ---
print("\nBuilding VGG16 model for transfer learning...")

# Load VGG16 pre-trained on ImageNet, excluding the top (classification) layer
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))

# Freeze the convolutional layers of VGG16
for layer in base_model.layers:
    layer.trainable = False

# Add custom classification layers on top of VGG16
x = base_model.output
x = Flatten()(x) # Flatten the output from the VGG16 convolutional base
x = Dense(256, activation='relu')(x) # A dense layer with ReLU activation
x = Dropout(0.5)(x) # Dropout for regularization to prevent overfitting
predictions = Dense(NUM_CLASSES, activation='softmax')(x) # Output layer with softmax for multi-class classification

# Create the full model
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer=Adam(learning_rate=LEARNING_RATE),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# --- 3. Train the Model ---
print("\nTraining the model...")

# Callbacks for saving the best model and early stopping
checkpoint = ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    epochs=NUM_EPOCHS,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // BATCH_SIZE,
    callbacks=[checkpoint, early_stopping]
)

print("\nModel training finished.")

# --- 4. Evaluate the Model ---
print("\nEvaluating the model on the validation set...")
val_loss, val_accuracy = model.evaluate(validation_generator, steps=validation_generator.samples // BATCH_SIZE)
print(f"Validation Loss: {val_loss:.4f}")
print(f"Validation Accuracy: {val_accuracy:.4f}")

# --- 5. Plot Training History ---
print("\nPlotting training history...")
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

print("\nModel training and evaluation script finished.")