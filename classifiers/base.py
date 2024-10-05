import json

class BaseClassifier:
    def __init__(self):
        self.cleaned_speeches = None
        self.labels = None

    # Method to load preprocessed data
    def load_preprocessed_data(self, filename='./speeches/preprocessed_speeches.json'):
        with open(filename, 'r') as f:
            data = json.load(f)
        self.cleaned_speeches = data['speeches']
        self.labels = data['labels']
        print("Data loaded successfully.")

    # Placeholder for a fit method (to be implemented in child classes)
    def fit(self, X, y):
        raise NotImplementedError("Subclasses should implement this!")

    # Placeholder for a predict method (to be implemented in child classes)
    def predict(self, X):
        raise NotImplementedError("Subclasses should implement this!")