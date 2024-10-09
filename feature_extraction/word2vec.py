# word2vec.py
import numpy as np
import gensim.downloader as api
from sklearn.model_selection import train_test_split
from .base import BaseFeatureExtractor

class Word2VecExtractor(BaseFeatureExtractor):
    def __init__(self):
        self.model = api.load('word2vec-google-news-300')
        # TODO: Word2Vec model itself is not a vectorizer, but we'll manage feature extraction here for now
        super().__init__(vectorizer=None)

    def vectorize_speech(self, speech):
        """Vectorize a single speech using the Word2Vec model."""
        words = speech.split()  # Assuming speech is already tokenized
        word_vectors = [self.model[word] for word in words if word in self.model]
        if len(word_vectors) == 0:
            return np.zeros(300)  # 300 is the dimension of Word2Vec
        return np.mean(word_vectors, axis=0)

    def extract(self, cleaned_speeches, labels):
        """Extract Word2Vec features."""
        X_word2vec = np.array([self.vectorize_speech(speech) for speech in cleaned_speeches])
        X_train_word2vec, X_test_word2vec, y_train, y_test = train_test_split(X_word2vec, labels, test_size=0.2, random_state=42)
        return X_train_word2vec, X_test_word2vec, y_train, y_test