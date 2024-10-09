import json
import joblib
import os
from sklearn.metrics import accuracy_score, classification_report
from feature_extraction.tfidf import TfidfExtractor
from feature_extraction.bow import BagOfWordsExtractor
from feature_extraction.word2vec import Word2VecExtractor
from feature_extraction.ngrams import NgramsExtractor
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

def load_preprocessed_data(filename='./speeches/preprocessed_speeches.json'):
    """Load preprocessed speeches and labels from a JSON file."""
    with open(filename, 'r') as f:
        data = json.load(f)
    print('Speeches and labels loaded successfully.')
    return data['speeches'], data['labels']

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

def main():
    # Load preprocessed data
    cleaned_speeches, labels = load_preprocessed_data()

    # Create necessary directories for saving models
    os.makedirs('classifiers', exist_ok=True)
    os.makedirs('vectorizers', exist_ok=True)

    # Create instances of feature extractors
    feature_extractors = [
        TfidfExtractor(),
        BagOfWordsExtractor(),
        Word2VecExtractor(),
        NgramsExtractor(),
    ]

    # Loop through each feature extraction method
    for extractor in feature_extractors:
        print(f"\nEvaluating {extractor.__class__.__name__}...")
        X_train, X_test, y_train, y_test = extractor.extract(cleaned_speeches, labels)

        # Save the vectorizer if applicable
        if extractor.vectorizer is not None:
            vectorizer_filename = f"vectorizers/{extractor.__class__.__name__}_vectorizer.pkl"
            extractor.save_vectorizer(vectorizer_filename)
            print(f"Saved vectorizer to {vectorizer_filename}")

        # Train and evaluate classifiers on the extracted features
        train_and_evaluate_classifiers(X_train, X_test, y_train, y_test, extractor.__class__.__name__)

if __name__ == '__main__':
    main()