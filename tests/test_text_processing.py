import unittest
import pandas as pd
from src.text import basic_language_detect, simple_preprocess_text, augment_text, preprocess_text_data

class TestTextProcessing(unittest.TestCase):

    def test_basic_language_detect(self):
        # Test correct language detection
        self.assertEqual(basic_language_detect("This is an English sentence."), 'en')
        self.assertEqual(basic_language_detect("C'est une phrase en français."), 'fr')
        self.assertEqual(basic_language_detect("Esta es una oración en español."), 'es')
        self.assertEqual(basic_language_detect("Ein Satz auf Deutsch."), 'unknown')

    def test_simple_preprocess_text(self):
        # Test text preprocessing, stopword removal, and tokenization
        text = "This is a simple test sentence with stopwords."
        result = simple_preprocess_text(text)
        expected = "this simple test sentence with stopwords"
        self.assertEqual(result, expected)

    def test_augment_text(self):
        # Test text augmentation using SynonymAug
        text = "The quick brown fox jumps over the lazy dog."
        result = augment_text(text)
        self.assertIsInstance(result, list)
        self.assertTrue(len(result) > 0)

    def test_preprocess_text_data(self):
        # Test text data preprocessing in DataFrame
        data = {'designation': ['Test1', 'Test2', None], 'description': ['Description1', 'Description2', None]}
        X_train = pd.DataFrame(data)
        preprocess_text_data(X_train)
        self.assertEqual(X_train['designation'].isnull().sum(), 0)
        self.assertEqual(X_train['description'].isnull().sum(), 0)

if __name__ == '__main__':
    unittest.main()