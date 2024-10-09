import joblib

class BaseFeatureExtractor:
    def __init__(self, vectorizer):
        self.vectorizer = vectorizer

    def save_vectorizer(self, path):
        """Save the vectorizer to a file."""
        joblib.dump(self.vectorizer, path)

    def extract(self, cleaned_speeches, labels):
        """Extract features from cleaned speeches. To be overridden by subclasses."""
        raise NotImplementedError("Subclasses should implement this method.")