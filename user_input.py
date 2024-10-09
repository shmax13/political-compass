import joblib
import os
import itertools
from collections import Counter

def load_model(model_path):
    """Load the trained model from a file."""
    model = joblib.load(model_path)
    return model

def load_input_file(file_path):
    """Read input text from a file."""
    with open(file_path, 'r') as file:
        text = file.read()
    return text

def classify_input(model, vectorizer, input_text):
    """Classify the input text using the provided model."""
    # transform and predict
    features = vectorizer.transform([input_text])  
    prediction = model.predict(features)
    return prediction[0]  

def main():
    classifiers = ['Logistic_Regression', 'Random_Forest_Classifier', 'Support_Vector_Classifier']
    vectorizers = ['TfidfExtractor', 'BagOfWordsExtractor', 'NgramsExtractor'] # TODO w2vec

    # Store predictions for final majority voting
    all_predictions = []

    # Loop through all combinations of classifiers and vectorizers
    for extractor_name, classifier_name in itertools.product(vectorizers, classifiers):
        # Construct paths for the model and vectorizer
        model_path = f'classifiers/{extractor_name}_{classifier_name}.pkl'
        vectorizer_path = f'vectorizers/{extractor_name}_vectorizer.pkl'

        # Check if both model and vectorizer exist before loading
        if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
            print(f"Skipping: {model_path} or {vectorizer_path} does not exist.")
            continue

        # Load the pre-trained model and vectorizer
        model = load_model(model_path)
        vectorizer = load_model(vectorizer_path)

        # Load input text
        input_file_path = 'speeches/single/input.txt'  
        input_text = load_input_file(input_file_path)

        # Classify the input text
        prediction = classify_input(model, vectorizer, input_text)
        all_predictions.append(prediction) 

        print(f"Prediction for {extractor_name} and {classifier_name}: {prediction}")

    # Calculate the majority vote
    final_prediction = Counter(all_predictions).most_common(1)[0][0] 
    print(f"Final prediction (majority vote): {final_prediction}")

if __name__ == '__main__':
    main()