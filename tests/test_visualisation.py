import unittest
import numpy as np
import os
import sys
from unittest.mock import patch, MagicMock
import gc

# Добавляем путь к папке 'src', где находится visualisation.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

class TestVisualisation(unittest.TestCase):
    
    @patch('gc.collect')
    def test_free_up_memory(self, mock_gc_collect):
        from visualisation import free_up_memory
        free_up_memory()
        mock_gc_collect.assert_called_once()

    @patch('numpy.load')
    def test_load_data(self, mock_np_load):
        base_path = 'data'
        mock_np_load.side_effect = [np.zeros((100, 300)), np.zeros((100, 2048)), np.zeros((100, 27))]
        from visualisation import load_data
        X_train_text, train_image_features_balanced, y_train = load_data()
        self.assertEqual(X_train_text.shape, (100, 300))
        self.assertEqual(train_image_features_balanced.shape, (100, 2048))
        self.assertEqual(y_train.shape, (100, 27))
        
    @patch('visualisation.train_test_split')
    def test_train_test_split(self, mock_train_test_split):
        X_train_text = np.zeros((100, 300))
        train_image_features_balanced = np.zeros((100, 2048))
        y_train = np.zeros((100, 27))
        
        mock_train_test_split.return_value = (
            np.zeros((90, 300)), np.zeros((10, 300)),
            np.zeros((90, 2048)), np.zeros((10, 2048)),
            np.zeros((90, 27)), np.zeros((10, 27))
        )
        
        from visualisation import split_data
        X_train_text, X_test_text, train_image_features_balanced, test_image_features, y_train, y_test = split_data(
            X_train_text, train_image_features_balanced, y_train)
        self.assertEqual(X_train_text.shape, (90, 300))
        self.assertEqual(X_test_text.shape, (10, 300))
        self.assertEqual(train_image_features_balanced.shape, (90, 2048))
        self.assertEqual(test_image_features.shape, (10, 2048))
        self.assertEqual(y_train.shape, (90, 27))
        self.assertEqual(y_test.shape, (10, 27))

if __name__ == '__main__':
    unittest.main()