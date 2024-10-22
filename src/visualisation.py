import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import gc
from sklearn import metrics
from sklearn.model_selection import train_test_split
import os

# Function to free up memory
def free_up_memory():
    gc.collect()

# Function to load data
def load_data(base_path='data'):
    X_train_text = np.load(os.path.join(base_path, 'X_train_tfidf_all.npy'))
    train_image_features_balanced = np.load(os.path.join(base_path, 'train_image_features.npy'))
    y_train = np.load(os.path.join(base_path, 'Y_train_encoded_all.npy'))
    return X_train_text, train_image_features_balanced, y_train

# Function to split data into train and test sets
def split_data(X_train_text, train_image_features_balanced, y_train):
    X_train_text, X_test_text, train_image_features_balanced, test_image_features, y_train, y_test = train_test_split(
        X_train_text, train_image_features_balanced, y_train, test_size=0.1, random_state=42
    )
    return X_train_text, X_test_text, train_image_features_balanced, test_image_features, y_train, y_test

if __name__ == "__main__":
    # Load and split data
    X_train_text, train_image_features_balanced, y_train = load_data()
    X_train_text, X_test_text, train_image_features_balanced, test_image_features, y_train, y_test = split_data(
        X_train_text, train_image_features_balanced, y_train
    )

    # Assuming number_to_cat is a dictionary that maps class numbers to category labels
    number_to_cat = {
        '0': "10", '1': "40", '2': "50", '3': "60", '4': "1140", '5': "1160",
        '6': "1180", '7': "1280", '8': "1281", '9': "1300", '10': "1301",
        '11': "1302", '12': "1320", '13': "1560", '14': "1920", '15': "1940",
        '16': "2060", '17': "2220", '18': "2280", '19': "2403", '20': "2462",
        '21': "2522", '22': "2582", '23': "2583", '24': "2585", '25': "2705",
        '26': "2905", 'macro avg': 'macro avg', 'weighted avg': 'weighted avg'
    }

    # Load the trained model
    model = tf.keras.models.load_model('data/simple_model.keras')

    # Evaluate model on training set
    Y_train_pred = model.predict([X_train_text, train_image_features_balanced], batch_size=32)
    Y_train_pred_classes = np.argmax(Y_train_pred, axis=1)
    print("Training set accuracy:")
    print(f"Accuracy: {accuracy_score(y_train, Y_train_pred_classes) * 100:.2f}%")
    free_up_memory()

    # Make predictions on the test set
    Y_test_pred = model.predict([X_test_text, test_image_features])
    Y_test_pred_classes = np.argmax(Y_test_pred, axis=1)

    # Calculate accuracy on test set
    accuracy = accuracy_score(y_test, Y_test_pred_classes)
    print(f"Accuracy on test set: {accuracy * 100:.2f}%")

    # Generate classification report for test set
    report = metrics.classification_report(y_test, Y_test_pred_classes, output_dict=True)
    print("Test set classification report:")
    print(metrics.classification_report(y_test, Y_test_pred_classes))

    # Extract precision, recall, and F1-score for each class
    precision = {key: value['precision'] for key, value in report.items() if isinstance(value, dict)}
    recall = {key: value['recall'] for key, value in report.items() if isinstance(value, dict)}
    f1score = {key: value['f1-score'] for key, value in report.items() if isinstance(value, dict)}

    # Adjust labels
    cat_labels = np.array([number_to_cat[str(number)] for number in list(precision.keys())])

    # Create a bar plot for precision, recall, and F1-score
    labels = cat_labels
    precision_values = np.array(list(precision.values()))
    recall_values = np.array(list(recall.values()))
    f1score_values = np.array(list(f1score.values()))

    x = np.arange(len(labels))  # the label locations
    width = 0.2  # the width of the bars

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.grid()
    rects1 = ax.bar(x - width, precision_values, width, label='Precision')
    rects2 = ax.bar(x, recall_values, width, label='Recall')
    rects3 = ax.bar(x + width, f1score_values, width, label='F1-Score')

    # Add some text for labels, title, and custom x-axis tick labels, etc.
    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Scores', fontsize=12)
    ax.set_xticks(x)
    ax.set_ylim(0.0, 1.0)
    ax.set_xticklabels(labels, rotation=90, ha='right', fontsize=10)
    ax.tick_params(axis='y', labelsize=10)
    ax.legend(loc='upper right')
    ax.set_title("Classification Report", fontsize=14)

    fig.tight_layout()
    plt.show()

    # Confusion matrix for test set
    conf_matrix = confusion_matrix(y_test, Y_test_pred_classes)
    plt.figure(figsize=(14, 10))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

    # Plot confusion matrix with normalized values
    cm_cat_labels = cat_labels[:27]

    cm = metrics.confusion_matrix(y_test, Y_test_pred_classes, normalize='true')
    disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=cm_cat_labels)
    fig, ax = plt.subplots(figsize=(14, 14))
    disp.plot(ax=ax, colorbar=False)

    plt.xlabel('Predicted Labels', fontsize=20)
    plt.ylabel('True Labels', fontsize=20)
    plt.xticks(rotation=90, fontsize=16)
    plt.yticks(fontsize=16)
    plt.title("Confusion Matrix", fontsize=24)

    # Format annotations to show up to three decimal places
    for text in ax.texts:
        text.set_text(f'{float(text.get_text()):.2f}')

    plt.show()

    # Plotting distribution of original and predicted labels on test set
    plt.figure(figsize=(14, 7))

    # Diagram of the original labels distribution
    plt.subplot(1, 2, 1)
    original_counts = pd.DataFrame(y_test).value_counts()
    plt.pie(original_counts,
            labels=original_counts.index.to_numpy(),
            autopct='%1.2f%%',
            pctdistance=0.88)
    plt.title('Fractions of product types in y_test')

    # Diagram of the predicted labels distribution
    plt.subplot(1, 2, 2)
    predicted_counts = pd.DataFrame(Y_test_pred_classes).value_counts()
    plt.pie(predicted_counts,
            labels=predicted_counts.index.to_numpy(),
            autopct='%1.2f%%',
            pctdistance=0.88)
    plt.title('Fractions of product types in Y_test_pred_classes')

    plt.show()

    # Freeing up memory after displaying the confusion matrix
    free_up_memory()