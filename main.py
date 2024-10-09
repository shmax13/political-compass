import json
from feature_extraction.tfidf import extract_tfidf
from feature_extraction.bow import extract_bow
from feature_extraction.word2vec import extract_word2vec
from feature_extraction.ngrams import extract_ngrams
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Function to load preprocessed speeches
def load_preprocessed_data(filename='./speeches/preprocessed_speeches.json'):
    with open(filename, 'r') as f:
        data = json.load(f)
    print('Speeches and labels loaded successfully.')
    return data['speeches'], data['labels']

# Main function
def main():
    # Load preprocessed data
    cleaned_speeches, labels = load_preprocessed_data()

    # Create a list of classifiers
    classifiers = {
        'Logistic Regression': LogisticRegression(),
        'Support Vector Classifier': SVC(),
        'Random Forest Classifier': RandomForestClassifier()
    }

    # Dictionary of feature extraction methods
    extraction_methods = {
        'TF-IDF': extract_tfidf,
        'Bag of Words (BoW)': extract_bow,
        'Word2Vec': extract_word2vec,
        'N-grams (Bigrams)': extract_ngrams
    }

    # Loop through each feature extraction method
    for method_name, extraction_function in extraction_methods.items():
        print(f"\nEvaluating {method_name}...")
        X_train, X_test, y_train, y_test = extraction_function(cleaned_speeches, labels)
        
        # Loop through each classifier
        for name, classifier in classifiers.items():
            print(f"Training {name}...")
            classifier.fit(X_train, y_train)
            predictions = classifier.predict(X_test)
            accuracy = accuracy_score(y_test, predictions)
            report = classification_report(y_test, predictions, zero_division=0)
            print(f"Accuracy ({method_name}) with {name}: {accuracy * 100:.2f}%")
            print("Classification Report:")
            print(report)

if __name__ == '__main__':
    main()