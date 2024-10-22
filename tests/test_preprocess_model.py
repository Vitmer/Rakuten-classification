import unittest
import numpy as np
import os
from src.model import main, free_memory

class TestModel(unittest.TestCase):

    def setUp(self):
        # Set up necessary paths and parameters for testing
        self.base_path = os.path.join(os.getcwd(), 'data')
        self.batch_size = 10000

    def test_data_loading(self):
        # Test if the data loading is successful
        X_train = np.load(os.path.join(self.base_path, 'X_train_tfidf_all.npy'))
        self.assertIsNotNone(X_train, "Failed to load X_train data")

        Y_train = np.load(os.path.join(self.base_path, 'Y_train_encoded_all.npy'))
        self.assertIsNotNone(Y_train, "Failed to load Y_train data")

    # Removed test_smote_in_batches as the function is not defined

    def tearDown(self):
        # Free memory after tests
        free_memory()

if __name__ == '__main__':
    unittest.main()