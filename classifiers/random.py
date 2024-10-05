import random
from classifiers.base import BaseClassifier

class RandomClassifier(BaseClassifier):
    def __init__(self):
        super().__init__()

    def fit(self, X, y):
        # do nothing
        pass

    def predict(self, X):
        # two classes: Left-Leaning and Right-Leaning
        possible_labels = ['Left-Leaning', 'Right-Leaning']
        predictions = [random.choice(possible_labels) for _ in range(X.shape[0])]  
        return predictions