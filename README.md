RAKUTEN-CLASSIFICATION

Overview

RAKUTEN-CLASSIFICATION is a machine learning project that focuses on the preprocessing and classification of text and image data, combining text feature extraction with image feature extraction. The project includes separate modules for handling both types of data, and an extensive visualization of model performance, including accuracy, precision, and other key metrics.

Project Structure

The project is organized as follows:

	•	src: This directory contains the main source code for data preprocessing, model building, and visualization.
	•	image.py: Handles image preprocessing and feature extraction.
	•	model.py: Builds and trains the classification model using the combined features from both text and images.
	•	text.py: Responsible for text preprocessing and feature extraction, including cleaning, tokenization, and vectorization.
	•	visualisation.py: Provides functions to visualize the model performance, including confusion matrices and accuracy/loss plots.
	•	tests: Contains unit tests to validate the functionality of each module.
	•	test_preprocess_image.py: Unit tests for image preprocessing.
	•	test_preprocess_model.py: Unit tests for model-related operations.
	•	test_text_processing.py: Unit tests for text preprocessing.
	•	test_visualisation.py: Unit tests for model performance visualization.
	•	data: Directory for storing input and processed data.

Key Features

	•	Text Preprocessing:
	•	Cleaning and tokenizing the input text data.
	•	Feature extraction using TF-IDF and other techniques.
	•	Image Preprocessing:
	•	Rescaling, augmentation, and feature extraction using deep learning models.
	•	Model Training:
	•	Combines both text and image features to train a classification model.
	•	Includes techniques such as class balancing and normalization.
	•	Visualization:
	•	Model evaluation through confusion matrices, classification reports, and loss/accuracy plots.

Installation

To run the project, you need to install the required dependencies:

pip install -r requirements.txt

Running the Project

	1.	Preprocessing: Start by running the preprocessing scripts for both text and image data:
	•	image.py for image preprocessing.
	•	text.py for text preprocessing.
	2.	Model Training: Train the model by running model.py.
	3.	Visualization: After the model is trained, use visualisation.py to analyze the performance.

Testing

Run the unit tests using the following command:

python -m unittest discover tests

Contributing

Feel free to open issues and create pull requests if you’d like to contribute to the project.

License

This project is licensed under the MIT License.
