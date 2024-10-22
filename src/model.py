import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Nadam
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import gc
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set seeds for reproducibility
seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)
logger.info('Seeds set for reproducibility')

# Function to free up memory
def free_memory():
    gc.collect()
    logger.info('Memory freed')

# Define the base path for data
base_path = os.path.join(os.getcwd(), 'data')
logger.info(f'Data directory set to: {base_path}')

def main():
    # Load text features
    try:
        X_train_tfidf = np.load(os.path.join(base_path, 'X_train_tfidf_all.npy'))
        Y_train_encoded = np.load(os.path.join(base_path, 'Y_train_encoded_all.npy'))
        logger.info('Text features loaded successfully')
    except Exception as e:
        logger.error(f'Error loading text features: {e}')

    # Load image features
    try:
        train_image_features = np.load(os.path.join(base_path, 'train_image_features.npy'))
        logger.info('Image features loaded successfully')
    except Exception as e:
        logger.error(f'Error loading image features: {e}')

    # Encode labels
    try:
        label_encoder = LabelEncoder()
        Y_train_encoded = label_encoder.fit_transform(Y_train_encoded)
        logger.info('Labels encoded successfully')
    except Exception as e:
        logger.error(f'Error encoding labels: {e}')

    # Free up memory
    free_memory()

    # Balance dataset using SMOTE in batches
    smote = SMOTE(random_state=42)
    batch_size = 10000  # Batch size for SMOTE

    def smote_in_batches(X, y, image_features, batch_size, output_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            logger.info(f'Created directory: {output_dir}')
        
        X_resampled = []
        y_resampled = []
        image_features_resampled = []
        
        for start in range(0, len(X), batch_size):
            end = min(start + batch_size, len(X))
            X_batch, y_batch = X[start:end], y[start:end]
            image_batch = image_features[start:end]
            X_res, y_res = smote.fit_resample(X_batch, y_batch)
            
            # Calculate the number of new samples added by SMOTE
            new_samples_count = len(X_res) - len(X_batch)
            if new_samples_count > 0:
                new_indices = np.random.choice(len(image_batch), new_samples_count, replace=True)
                image_res = np.concatenate([image_batch, image_batch[new_indices]])
            else:
                image_res = image_batch
            
            # Save the resampled batches to disk
            np.save(os.path.join(output_dir, f'X_resampled_{start}_{end}.npy'), X_res)
            np.save(os.path.join(output_dir, f'y_resampled_{start}_{end}.npy'), y_res)
            np.save(os.path.join(output_dir, f'image_resampled_{start}_{end}.npy'), image_res)
            logger.info(f'Saved resampled batch {start} to {end}')

    output_dir = os.path.join(base_path, 'resampled_batches')
    smote_in_batches(X_train_tfidf, Y_train_encoded, train_image_features, batch_size, output_dir)

    # Free up memory
    free_memory()

    # Load all resampled batches from disk and concatenate them
    logger.info('Loading and concatenating resampled batches')
    X_train_tfidf_balanced = np.vstack([np.load(os.path.join(output_dir, f'X_resampled_{start}_{end}.npy')) for start, end in 
                                        [(i, min(i + batch_size, len(X_train_tfidf))) for i in range(0, len(X_train_tfidf), batch_size)]])
    Y_train_encoded_balanced = np.hstack([np.load(os.path.join(output_dir, f'y_resampled_{start}_{end}.npy')) for start, end in 
                                          [(i, min(i + batch_size, len(X_train_tfidf))) for i in range(0, len(X_train_tfidf), batch_size)]])
    train_image_features_balanced = np.vstack([np.load(os.path.join(output_dir, f'image_resampled_{start}_{end}.npy')) for start, end in 
                                               [(i, min(i + batch_size, len(X_train_tfidf))) for i in range(0, len(X_train_tfidf), batch_size)]])
    logger.info('Concatenation complete')

    # Split balanced data into 90% (train + validation) and 10% (test)
    logger.info('Splitting data into train/validation and test sets')
    X_train_val_text, X_test_text, y_train_val, y_test = train_test_split(X_train_tfidf_balanced, Y_train_encoded_balanced, test_size=0.1, random_state=42)
    train_val_image_features, test_image_features = train_test_split(train_image_features_balanced, test_size=0.1, random_state=42)

    # Check sizes
    assert len(X_train_val_text) == len(train_val_image_features), f"Size mismatch: X_train_val_text ({len(X_train_val_text)}) and train_val_image_features ({len(train_val_image_features)})"
    assert len(X_test_text) == len(test_image_features), f"Size mismatch: X_test_text ({len(X_test_text)}) and test_image_features ({len(test_image_features)})"
    logger.info('Data split successfully')

    # Free up memory
    free_memory()

    # Split 90% data into 80% (train) and 20% (validation)
    logger.info('Splitting data into train and validation sets')
    X_train_text, X_val_text, y_train, y_val = train_test_split(X_train_val_text, y_train_val, test_size=0.2, random_state=42)
    train_image_features_balanced, val_image_features = train_test_split(train_val_image_features, test_size=0.2, random_state=42)

    # Check sizes
    assert len(X_train_text) == len(train_image_features_balanced), f"Size mismatch: X_train_text ({len(X_train_text)}) and train_image_features_balanced ({len(train_image_features_balanced)})"
    assert len(X_val_text) == len(val_image_features), f"Size mismatch: X_val_text ({len(X_val_text)}) and val_image_features ({len(val_image_features)})"
    logger.info('Train and validation data split successfully')

    # Free up memory
    free_memory()

    # Define model
    def build_model(input_shape_text, input_shape_image):
        logger.info('Building model')
        text_input = Input(shape=(input_shape_text,), name='text_input')
        x1 = Dense(512, activation='relu')(text_input)
        x1 = BatchNormalization()(x1)
        x1 = Dropout(0.5)(x1)
        x1 = Dense(256, activation='relu')(x1)
        x1 = BatchNormalization()(x1)
        x1 = Dropout(0.5)(x1)

        image_input = Input(shape=(input_shape_image,), name='image_input')
        x2 = Dense(256, activation='relu')(image_input)
        x2 = BatchNormalization()(x2)
        x2 = Dropout(0.5)(x2)
        x2 = Dense(128, activation='relu')(x2)
        x2 = BatchNormalization()(x2)
        x2 = Dropout(0.5)(x2)

        combined = concatenate([x1, x2])

        x = Dense(64, activation='relu')(combined)
        x = BatchNormalization()(x)
        x = Dropout(0.25)(x)
        output = Dense(len(np.unique(Y_train_encoded)), activation='softmax')(x)

        model = Model(inputs=[text_input, image_input], outputs=output)
        model.compile(optimizer=Nadam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        logger.info('Model built and compiled')
        return model

    # Create and train model
    model = build_model(X_train_text.shape[1], train_image_features_balanced.shape[1])

    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

    # Free up memory
    free_memory()

    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights = {i: class_weights[i] for i in range(len(class_weights))}
    logger.info('Class weights computed')

    # Train model with balanced data
    logger.info('Training model')
    history = model.fit([X_train_text, train_image_features_balanced], y_train, 
                        epochs=30, batch_size=64, class_weight=class_weights, 
                        validation_data=([X_val_text, val_image_features], y_val), 
                        callbacks=[early_stopping, reduce_lr])

    logger.info('Model training complete')

    # Save trained model
    model_path = os.path.join(base_path, 'simple_model.keras')
    try:
        model.save(model_path)
        logger.info(f'Model saved successfully at {model_path}')
    except Exception as e:
        logger.error(f'Error saving model: {e}')

    # Free up memory after saving model
    free_memory()

    # Make predictions for the test data
    logger.info('Predicting on test data')
    y_pred = model.predict([X_test_text, test_image_features])
    y_pred_classes = np.argmax(y_pred, axis=1)

    # Decode labels
    y_test_decoded = label_encoder.inverse_transform(y_test)
    y_pred_decoded = label_encoder.inverse_transform(y_pred_classes)

    # Generate classification report
    report = classification_report(y_test_decoded, y_pred_decoded, target_names=[str(cls) for cls in label_encoder.classes_])
    logger.info('Classification Report:\n' + report)

    # Save the report to a file
    report_path = os.path.join(base_path, 'classification_report.txt')
    with open(report_path, 'w') as f:
        f.write('Classification Report:\n')
        f.write(report)
        logger.info(f'Classification report saved to {report_path}')

    logger.info('Process completed')

if __name__ == "__main__":
    main()