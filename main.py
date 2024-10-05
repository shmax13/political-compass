import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from classifiers.logistic_regression import LogisticRegressionClassifier
from classifiers.random import RandomClassifier

def load_preprocessed_data(filename='./speeches/preprocessed_speeches.json'):
    with open(filename, 'r') as f:
        data = json.load(f)
    print('Speeches and labels loaded successfully.')
    return data['speeches'], data['labels']

def main():
    # Load preprocessed data
    cleaned_speeches, labels = load_preprocessed_data()

    # Create a list of classifiers
    classifiers = [
        RandomClassifier(),
        LogisticRegressionClassifier()]

    # Feature extraction using TF-IDF
    vectorizer = TfidfVectorizer(max_features=5000)  # Limit features to avoid overfitting
    X = vectorizer.fit_transform(cleaned_speeches)
    y = labels

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    for classifier in classifiers:
        print(f"Training {classifier.__class__.__name__}...")

        # Train the classifier
        classifier.fit(X_train, y_train)

        # Make predictions on the test set
        predictions = classifier.predict(X_test)

        # Evaluate the classifier
        accuracy = accuracy_score(y_test, predictions)
        report = classification_report(y_test, predictions, zero_division=0)

        # Print the results
        print(f"Accuracy: {accuracy * 100:.2f}%")
        print("Classification Report:")
        print(report)

if __name__ == '__main__':
    main()