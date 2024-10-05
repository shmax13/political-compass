from sklearn.linear_model import LogisticRegression

from classifiers.base import BaseClassifier

class LogisticRegressionClassifier(BaseClassifier):
    def __init__(self):
        super().__init__()
        self.model = LogisticRegression()

    # Implement the fit method
    def fit(self, X, y):
        self.model.fit(X, y)
        print("Model trained successfully.")

    # Implement the predict method
    def predict(self, X):
        return self.model.predict(X)