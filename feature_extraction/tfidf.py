from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from .base import BaseFeatureExtractor
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler

class TfidfExtractor(BaseFeatureExtractor):
    def __init__(self, max_features=5000):
        # vectorizer = TfidfVectorizer(max_features=max_features)
        vectorizer = TfidfVectorizer(max_features=max_features, max_df=0.9, min_df=0.02, ngram_range=(1, 3))

        super().__init__(vectorizer)

        self.scaler = StandardScaler(with_mean=False)
        self.svd = TruncatedSVD(n_components=100)

    def extract(self, cleaned_speeches, labels):
        """Extract TF-IDF features and save the vectorizer."""
        X_tfidf = self.vectorizer.fit_transform(cleaned_speeches)
        X_train_tfidf, X_test_tfidf, y_train, y_test = train_test_split(X_tfidf, labels, test_size=0.2, random_state=42)
        return X_train_tfidf, X_test_tfidf, y_train, y_test