import unittest
from unittest.mock import patch, MagicMock
from src.image import preprocess_image, load_and_preprocess_image, build_image_model, extract_features_from_directory
import numpy as np
import tensorflow as tf
from PIL import Image

class TestImageProcessing(unittest.TestCase):

    @patch("src.image.ImageOps.fit")  # Patched ImageOps.fit instead of resize
    @patch("src.image.Image.open")
    def test_preprocess_image(self, mock_open, mock_fit):
        # Create a mock image and patch the open function
        mock_image = MagicMock(spec=Image.Image)
        mock_open.return_value = mock_image

        # Call the function
        result = preprocess_image("dummy_path", target_size=(224, 224))

        # Check if the image was opened and resized using ImageOps.fit
        mock_open.assert_called_once_with("dummy_path")
        mock_image.convert.assert_called_once_with("RGB")
        mock_fit.assert_called_once()  # Check that fit was called, not resize

        # Check that the result is a normalized numpy array
        self.assertIsInstance(result, np.ndarray)
        self.assertTrue((0 <= result).all() and (result <= 1).all())

    @patch("src.image.preprocess_image")
    def test_load_and_preprocess_image(self, mock_preprocess):
        # Mock the return of preprocess_image function
        mock_preprocess.return_value = np.random.rand(224, 224, 3)

        # Call the function
        result = load_and_preprocess_image("dummy_path")

        # Verify that the image preprocessing happened and result is a tensor
        mock_preprocess.assert_called_once_with("dummy_path")
        self.assertIsInstance(result, tf.Tensor)
        self.assertEqual(result.shape, (224, 224, 3))
        self.assertEqual(result.dtype, tf.float32)

    def test_build_image_model(self):
        # Call the function to build the model
        model = build_image_model(input_size=(224, 224, 3))

        # Check that model is an instance of keras Model
        self.assertIsNotNone(model)
        self.assertIsInstance(model, tf.keras.Model)

        # Check if the output shape is as expected
        self.assertEqual(model.output_shape, (None, 1280))  # EfficientNetB0 output feature size

@patch("src.image.tf.keras.Model.predict")
@patch("src.image.tqdm")
@patch("src.image.load_and_preprocess_image")
@patch("src.image.os.listdir")
def test_extract_features_from_directory(self, mock_listdir, mock_load_image, mock_tqdm, mock_predict):
    # Mock predict method of the model and the tqdm
    mock_predict.return_value = np.random.rand(1, 1280)
    mock_tqdm.return_value = iter(["dummy_image_path"])

    # Mock the directory listing to simulate image files
    mock_listdir.return_value = ["image1.jpg", "image2.jpg"]  # Ensure listdir returns non-empty results

    # Mock the image loading function to return a valid tensor (not None)
    mock_load_image.return_value = np.random.rand(224, 224, 3)

    # Create a mock model
    mock_model = MagicMock()
    mock_model.output_shape = (None, 1280)

    # Call the function with mocks
    extract_features_from_directory("dummy_dir", {"image1.jpg": "label1", "image2.jpg": "label2"}, mock_model, output_file="dummy_output.npy")

    # Assert that the model's predict method was called
    self.assertTrue(mock_predict.called, "predict() method was not called as expected")
    self.assertEqual(mock_predict.call_count, 2)  # Assert that predict was called twice (for two images)

    # Add additional logging to verify where the process is failing
    if not mock_predict.called:
        print("Predict function was not called. Check image processing or directory listing.")

if __name__ == "__main__":
    unittest.main()