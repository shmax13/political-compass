from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from .base import BaseFeatureExtractor

class BagOfWordsExtractor(BaseFeatureExtractor):
    def __init__(self, max_features=5000):
        vectorizer = CountVectorizer(max_features=max_features)
        super().__init__(vectorizer)

    def extract(self, cleaned_speeches, labels):
        """Extract Bag of Words features and save the vectorizer."""
        X_bow = self.vectorizer.fit_transform(cleaned_speeches)
        X_train_bow, X_test_bow, y_train, y_test = train_test_split(X_bow, labels, test_size=0.2, random_state=42)
        return X_train_bow, X_test_bow, y_train, y_test