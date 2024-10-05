class BaseClassifier:
    def __init__(self):
        self.cleaned_speeches = None
        self.labels = None

    # Placeholder for a fit method (to be implemented in child classes)
    def fit(self, X, y):
        raise NotImplementedError("Subclasses should implement this!")

    # Placeholder for a predict method (to be implemented in child classes)
    def predict(self, X):
        raise NotImplementedError("Subclasses should implement this!")