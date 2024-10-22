import pandas as pd
import numpy as np
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import gc
from tqdm import tqdm
import re
import nlpaug.augmenter.word as naw

# Setting up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

tqdm.pandas()

def basic_language_detect(text):
    if re.search(r'\b(le|la|les|un|une|des)\b', text, re.IGNORECASE):
        return 'fr'
    elif re.search(r'\b(the|is|and|of|in|to)\b', text, re.IGNORECASE):
        return 'en'
    elif re.search(r'\b(el|la|los|las|un|una|unos|unas)\b', text, re.IGNORECASE):
        return 'es'
    else:
        return 'unknown'

manual_stopwords = set("""
a about above after again against all am an and any are aren't as at be because been before being below between both but by can't cannot could couldn't did didn't do does doesn't doing don't down during each few for from further had hadn't has hasn't have haven't having he he'd he'll he's her here here's hers herself him himself his how how's i i'd i'll i'm i've if in into is isn't it it's its itself let's me more most mustn't my myself no nor
un une des du le la les et à en dans par pour au aux avec ce cette ces comme il elle ils elles que qui sur son sa ses leur leurs où donc ne pas ni non plus on nous vous votre vos c'est ce sont car même si mais ou où or ni soit ni puis
""".split())

def simple_preprocess_text(text):
    tokens = re.findall(r'\b\w\w+\b', text.lower())
    filtered_tokens = [word for word in tokens if word not in manual_stopwords and len(word) > 2]
    return ' '.join(filtered_tokens)

synonym_aug = naw.SynonymAug(aug_src='wordnet')

def augment_text(text):
    return synonym_aug.augment(text)

def preprocess_text_data(X_train):
    logging.info("Preprocessing text data for training set.")
    
    X_train['designation'] = X_train['designation'].fillna('')
    X_train['description'] = X_train['description'].fillna('')
    
    X_train['text'] = X_train['designation'] + ' ' + X_train['description']
    
    X_train['processed_text'] = X_train['text'].progress_apply(simple_preprocess_text)
    
    return X_train

def add_new_features(df):
    logging.info("Calculating additional features for dataset.")
    df['num_words_designation'] = df['designation'].apply(lambda x: len(str(x).split()) if pd.notnull(x) else 0)
    df['num_words_description'] = df['description'].apply(lambda x: len(str(x).split()) if pd.notnull(x) else 0)
    df['avg_word_length_designation'] = df['designation'].apply(lambda x: np.mean([len(word) for word in str(x).split()]) if pd.notnull(x) and len(str(x).split()) > 0 else 0)
    df['avg_word_length_description'] = df['description'].apply(lambda x: np.mean([len(word) for word in str(x).split()]) if pd.notnull(x) and len(str(x).split()) > 0 else 0)
    df['num_special_chars_designation'] = df['designation'].apply(lambda x: sum(not c.isalnum() for c in str(x)) if pd.notnull(x) else 0)
    df['num_special_chars_description'] = df['description'].apply(lambda x: sum(not c.isalnum() for c in str(x)) if pd.notnull(x) else 0)
    df['ratio_digits_designation'] = df['designation'].apply(lambda x: sum(c.isdigit() for c in str(x)) / len(str(x)) if pd.notnull(x) and len(str(x)) > 0 else 0)
    df['ratio_digits_description'] = df['description'].apply(lambda x: sum(c.isdigit() for c in str(x)) / len(str(x)) if pd.notnull(x) and len(str(x).split()) > 0 else 0)
    df['num_punctuation_designation'] = df['designation'].apply(lambda x: sum(c in '.,;:!?()[]{}' for c in str(x)) if pd.notnull(x) else 0)
    df['num_punctuation_description'] = df['description'].apply(lambda x: sum(c in '.,;:!?()[]{}' for c in str(x)) if pd.notnull(x) else 0)
    return df

if __name__ == "__main__":
    # Logging the start of loading the data
    logging.info("Loading training dataset.")
    X_train = pd.read_csv('data/X_train_update.csv')
    Y_train = pd.read_csv('data/Y_train_update.csv')

    gc.collect()

    # Adding label column
    logging.info("Adding image label columns.")
    X_train['label'] = 'image_' + X_train['imageid'].astype(str) + '_product_' + X_train['productid'].astype(str) + '.jpg'

    # Applying language detection
    logging.info("Detecting languages for designation and description fields.")
    X_train['designation_language'] = X_train['designation'].apply(basic_language_detect)
    X_train['description_language'] = X_train['description'].fillna('').apply(basic_language_detect)

    # Cleaning the designation and description
    logging.info("Cleaning designation and description fields.")
    X_train['designation_cleaned'] = X_train['designation'].apply(simple_preprocess_text)
    X_train['description_cleaned'] = X_train['description'].fillna('').apply(simple_preprocess_text)

    # Augmenting text data
    logging.info("Augmenting text data.")
    X_train['designation_aug'] = X_train['designation_cleaned'].progress_apply(augment_text)
    X_train['description_aug'] = X_train['description_cleaned'].progress_apply(augment_text)

    # Cleaning and merging datasets
    logging.info("Copying and cleaning dataset.")
    X_train_cleaned = X_train.copy()
    Y_train_cleaned = Y_train.copy()

    X_train_cleaned = X_train_cleaned.drop(columns=['Unnamed: 0'], errors='ignore')
    Y_train_cleaned = Y_train_cleaned.drop(columns=['Unnamed: 0'], errors='ignore')

    X_train_cleaned = X_train_cleaned.merge(Y_train_cleaned, left_index=True, right_index=True)

    X_train_cleaned = preprocess_text_data(X_train_cleaned)

    # Adding new features
    logging.info("Adding new features based on text properties.")
    X_train_cleaned['language'] = X_train_cleaned.apply(lambda row: f"{row['designation_language']}_{row['description_language']}", axis=1)
    X_train_cleaned['text_length'] = X_train_cleaned['text'].apply(len)

    X_train_cleaned = add_new_features(X_train_cleaned)

    # Saving processed data
    logging.info("Saving processed data arrays.")
    X_train_cleaned.to_csv('data/X_train_cleaned.csv', index=False)
    
    gc.collect()

    # Vectorizing text data
    logging.info("Vectorizing text data.")
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train_cleaned['processed_text'])

    # Combining additional features
    logging.info("Combining TF-IDF features with additional features.")
    additional_features = ['text_length', 'num_words_designation', 'num_words_description', 'avg_word_length_designation', 
                           'avg_word_length_description', 'num_special_chars_designation', 'num_special_chars_description', 
                           'ratio_digits_designation', 'ratio_digits_description', 'num_punctuation_designation', 'num_punctuation_description']

    X_train_additional = X_train_cleaned[additional_features].values
    X_train_final = np.hstack([X_train_tfidf.toarray(), X_train_additional])

    # Encoding labels
    logging.info("Encoding target labels.")
    label_encoder = LabelEncoder()
    Y_train_encoded = label_encoder.fit_transform(Y_train_cleaned['prdtypecode'])

    # Saving processed data
    logging.info("Saving processed data arrays.")
    np.save('data/X_train_tfidf_all.npy', X_train_tfidf.toarray())
    np.save('data/Y_train_encoded_all.npy', Y_train_encoded)

    logging.info("Data processing completed.")
    gc.collect()