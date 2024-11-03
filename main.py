import json
import joblib
import os
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, mean_absolute_error, r2_score
from feature_extraction.tfidf import TfidfExtractor
from feature_extraction.bow import BagOfWordsExtractor
from feature_extraction.word2vec import Word2VecExtractor
from feature_extraction.ngrams import NgramsExtractor
from feature_extraction.bert import BERTExtractor
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression

def load_preprocessed_data(filename='./speeches/preprocessed_speeches.json'):
    """Load preprocessed speeches and labels from a JSON file."""
    with open(filename, 'r') as f:
        data = json.load(f)
    print('Speeches and labels loaded successfully.')
    return data['speeches'], data['labels'], data['coordinates']

def train_and_evaluate_classifiers(X_train, X_test, y_train, y_test, extractor_name):
    """Train and evaluate classifiers on the extracted features."""
    classifiers = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Support Vector Classifier': SVC(),
        'Random Forest Classifier': RandomForestClassifier()
    }

    for name, classifier in classifiers.items():
        print(f"Training {name}...")
        classifier.fit(X_train, y_train)
        predictions = classifier.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        report = classification_report(y_test, predictions, zero_division=0)
        print(f"Accuracy with {name}: {accuracy * 100:.2f}%")
        print("Classification Report:")
        print(report)

        # Save the classifier
        classifier_filename = f"classifiers/{extractor_name}_{name.replace(' ', '_')}.pkl"
        joblib.dump(classifier, classifier_filename)
        print(f"Saved {name} classifier to {classifier_filename}")


from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib

def train_and_evaluate_regressors(X_train, X_test, y_train, y_test, extractor_name):
    """Train and evaluate regressors on the extracted features."""
    regressors = {
        'Linear Regression (x)': LinearRegression(),
        'Linear Regression (y)': LinearRegression(),
        'Random Forest Regressor (x)': RandomForestRegressor(),
        'Random Forest Regressor (y)': RandomForestRegressor(),
        'Support Vector Regressor (x)': SVR(),
        'Support Vector Regressor (y)': SVR()
    }

    y_train = np.array(y_train)
    y_test = np.array(y_test)

    # Split y into x and y coordinates
    y_train_x = y_train[:, 0]  # x-coordinates
    y_train_y = y_train[:, 1]  # y-coordinates
    y_test_x = y_test[:, 0]
    y_test_y = y_test[:, 1]

    for name, regressor in regressors.items():
        print(f"Training {name}...")

        # Train and predict for x or y based on the regressor name
        if 'x' in name:
            regressor.fit(X_train, y_train_x)
            predictions = regressor.predict(X_test)
            mse = mean_squared_error(y_test_x, predictions)
            r2 = r2_score(y_test_x, predictions)
            mae = mean_absolute_error(y_test_x, predictions)
        else:
            regressor.fit(X_train, y_train_y)
            predictions = regressor.predict(X_test)
            mse = mean_squared_error(y_test_y, predictions)
            r2 = r2_score(y_test_y, predictions)
            mae = mean_absolute_error(y_test_y, predictions)
        
        # Print metrics
        print(f"MSE with {name}: {mse:.4f}")
        print(f"R² with {name}: {r2:.4f}")
        print(f"MAE with {name}: {mae:.4f}")

        # Save the regressor
        regressor_filename = f"regressors/{extractor_name}_{name.replace(' ', '_')}.pkl"
        joblib.dump(regressor, regressor_filename)
        print(f"Saved {name} regressor to {regressor_filename}")
    

def main():
    # Load preprocessed data
    cleaned_speeches, labels, coordinates = load_preprocessed_data()

    # Create necessary directories for saving models
    os.makedirs('classifiers', exist_ok=True)
    os.makedirs('regressors', exist_ok=True)
    os.makedirs('vectorizers', exist_ok=True)

    # Create instances of feature extractors
    feature_extractors = [
        TfidfExtractor(),
        BagOfWordsExtractor(),
        NgramsExtractor(),
        Word2VecExtractor(),
        BERTExtractor(),
    ]

    # Loop through each feature extraction method
    for extractor in feature_extractors:
        print(f"\nEvaluating {extractor.__class__.__name__}...")
        
        # Extract features for both classification and regression
        X_train, X_test, y_train_classification, y_test_classification = extractor.extract(cleaned_speeches, labels)
        _, _, y_train_regression, y_test_regression = extractor.extract(cleaned_speeches, coordinates)

        # Save the vectorizer if applicable
        if extractor.vectorizer is not None:
            vectorizer_filename = f"vectorizers/{extractor.__class__.__name__}_vectorizer.pkl"
            extractor.save_vectorizer(vectorizer_filename)
            print(f"Saved vectorizer to {vectorizer_filename}")

        # Train and evaluate classifiers on the extracted features
        train_and_evaluate_classifiers(X_train, X_test, y_train_classification, y_test_classification, extractor.__class__.__name__)

        # Train and evaluate regressors on the extracted features (for coordinates)
        train_and_evaluate_regressors(X_train, X_test, y_train_regression, y_test_regression, extractor.__class__.__name__)

if __name__ == '__main__':
    main()