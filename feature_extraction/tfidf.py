from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from .base import BaseFeatureExtractor
from sklearn.decomposition import PCA

class TfidfExtractor(BaseFeatureExtractor):
    def __init__(self, max_features=5000):
        vectorizer = TfidfVectorizer(max_features=max_features)
        super().__init__(vectorizer)


    def extract(self, cleaned_speeches, labels):
        """Extract TF-IDF features and save the vectorizer."""
        X_tfidf = self.vectorizer.fit_transform(cleaned_speeches)

        # Apply PCA to reduce dimensionality
        pca = PCA(n_components=200)
        X_tfidf = pca.fit_transform(X_tfidf)

        X_train_tfidf, X_test_tfidf, y_train, y_test = train_test_split(X_tfidf, labels, test_size=0.2, random_state=42)
        return X_train_tfidf, X_test_tfidf, y_train, y_test