# ngrams.py
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from .base import BaseFeatureExtractor

class NgramsExtractor(BaseFeatureExtractor):
    def __init__(self, ngram_range=(2, 2), max_features=5000):
        vectorizer = CountVectorizer(ngram_range=ngram_range, max_features=max_features)
        super().__init__(vectorizer)

    def extract(self, cleaned_speeches, labels):
        """Extract n-grams features."""
        X_ngrams = self.vectorizer.fit_transform(cleaned_speeches)
        X_train_ngrams, X_test_ngrams, y_train, y_test = train_test_split(X_ngrams, labels, test_size=0.2, random_state=42)
        return X_train_ngrams, X_test_ngrams, y_train, y_test