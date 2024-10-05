from classifiers.logistic_regression import LogisticRegressionClassifier
from classifiers.random import RandomClassifier  # Import the RandomClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

def main():
    # List of classifiers
    classifiers = [
        RandomClassifier(),
        LogisticRegressionClassifier()
    ]

    # Load preprocessed data using the base class method
    for classifier in classifiers:
        classifier.load_preprocessed_data()

        # Feature extraction using TF-IDF
        vectorizer = TfidfVectorizer(max_features=5000)  # Limit features to avoid overfitting
        X = vectorizer.fit_transform(classifier.cleaned_speeches)
        y = classifier.labels

        # Split the data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train the classifier
        classifier.fit(X_train, y_train)

        # Make predictions on the test set
        predictions = classifier.predict(X_test)

        # Evaluate the classifier
        accuracy = accuracy_score(y_test, predictions)
        report = classification_report(y_test, predictions)

        # Print the results for each classifier
        print(f"Classifier: {classifier.__class__.__name__}")
        print(f"Accuracy: {accuracy * 100:.2f}%")
        print("Classification Report:")
        print(report)

if __name__ == '__main__':
    main()