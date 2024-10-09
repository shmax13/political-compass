# feature_extraction/word2vec.py

import numpy as np
import gensim.downloader as api
from sklearn.model_selection import train_test_split

# Function to vectorize speeches using Word2Vec
def vectorize_speech(speech, model):
    words = speech.split()  # Assuming speech is already tokenized
    word_vectors = [model[word] for word in words if word in model]
    if len(word_vectors) == 0:
        return np.zeros(300)  # 300 is the dimension of Word2Vec
    return np.mean(word_vectors, axis=0)

def extract_word2vec(cleaned_speeches, labels):
    word2vec_model = api.load('word2vec-google-news-300')
    X_word2vec = np.array([vectorize_speech(speech, word2vec_model) for speech in cleaned_speeches])
    X_train_word2vec, X_test_word2vec, y_train, y_test = train_test_split(X_word2vec, labels, test_size=0.2, random_state=42)
    return X_train_word2vec, X_test_word2vec, y_train, y_test