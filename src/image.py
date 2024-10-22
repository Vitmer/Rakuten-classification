import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tqdm import tqdm
import os
from PIL import Image, ImageOps
import numpy as np
import gc
import re
import pandas as pd
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Paths to image directories
train_image_dir = 'data/images/image_train'

X_train_cleaned = pd.read_csv('data/X_train_cleaned.csv')

# Converting labels to dictionaries for easy lookup
train_labels = dict(zip(X_train_cleaned['imageid'].astype(str), X_train_cleaned['productid'].astype(str)))

# Function for preprocessing an image using PIL
def preprocess_image(image_path, target_size=(224, 224)):
    try:
        image = Image.open(image_path).convert('RGB')
        image = ImageOps.fit(image, target_size, Image.LANCZOS)
        image = np.array(image) / 255.0
        return image
    except Exception as e:
        logging.error(f"Error processing image {image_path}: {e}")
        return None

# Function for preprocessing and augmenting an image using TensorFlow, returning float32 tensors
def load_and_preprocess_image(image_path):
    image = preprocess_image(image_path)
    if image is None:
        return None
    image = tf.convert_to_tensor(image, dtype=tf.float32)
    
    # Apply augmentations
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
    
    image = tf.keras.applications.efficientnet.preprocess_input(image)
    return image

# Function for extracting image features one by one
def extract_features_from_directory(image_dir, labels, model, output_file='data/train_image_features.npy'):
    logging.info(f"Extracting features from directory: {image_dir}")
    image_paths = []
    indices = []  # Store the corresponding indices for each image
    for f in os.listdir(image_dir):
        if f.endswith('.jpg'):
            match = re.match(r'image_(\d+)_product_(\d+)\.jpg', f)
            if match:
                image_id, product_id = match.groups()
                if image_id in labels and labels[image_id] == product_id:
                    image_paths.append(os.path.join(image_dir, f))
                    indices.append(int(image_id))

    if not image_paths:
        logging.warning(f"No images found in directory: {image_dir}")
        return

    num_images = len(image_paths)
    features_shape = model.output_shape[1]
    all_features = np.zeros((num_images, features_shape), dtype=np.float32)  # Placeholder for all features
    
    feature_idx = 0
    for image_path in tqdm(image_paths, desc="Extracting features from images"):
        try:
            image = load_and_preprocess_image(image_path)
            if image is not None:
                image = tf.expand_dims(image, 0)  # Add batch dimension
                image_features = model.predict(image)
                all_features[feature_idx] = image_features[0]
                feature_idx += 1
                gc.collect()
        except Exception as e:
            logging.error(f"Error during feature extraction for image {image_path}: {e}")
    
    # Save features with correct indices
    sorted_indices = np.argsort(indices)
    all_features_sorted = all_features[sorted_indices]
    np.save(output_file, all_features_sorted)
    logging.info(f"Features saved to {output_file}")

# Function for building and compiling the EfficientNetB0 model
def build_image_model(input_size=(224, 224, 3)):
    try:
        base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=input_size)
        x = GlobalAveragePooling2D()(base_model.output)
        model = Model(inputs=base_model.input, outputs=x)
        return model
    except Exception as e:
        logging.error(f"Error building model: {e}")
        return None

# Ensure that the code only runs when the script is executed directly
if __name__ == "__main__":
    # Build and compile the model
    image_model = build_image_model(input_size=(224, 224, 3))

    if image_model is not None:
        # Extract image features for the training set
        extract_features_from_directory(train_image_dir, train_labels, image_model, output_file='data/train_image_features.npy')
        gc.collect()

    logging.info("Feature extraction for training set completed and saved to disk.")