import joblib
import os
import itertools
from collections import Counter
import gradio as gr

def load_model(model_path):
    """Load the trained model from a file."""
    model = joblib.load(model_path)
    return model

def classify_input(model, vectorizer, input_text):
    """Classify the input text using the provided model."""
    # Transform and predict
    features = vectorizer.transform([input_text])  
    prediction = model.predict(features)
    return prediction[0]  

def make_prediction(input_text):
    """Make predictions based on user input."""
    classifiers = ['Logistic_Regression', 'Random_Forest_Classifier', 'Support_Vector_Classifier']
    vectorizers = ['TfidfExtractor', 'BagOfWordsExtractor', 'NgramsExtractor']

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

        # Classify the input text
        prediction = classify_input(model, vectorizer, input_text)
        all_predictions.append(prediction) 

    # Calculate the majority vote
    final_prediction = Counter(all_predictions).most_common(1)[0][0] 
    return final_prediction

def main():
    # Create a Gradio interface
    iface = gr.Interface(
        fn=make_prediction, 
        inputs=gr.Textbox(label="Input Text"), 
        outputs=gr.Label(label="Prediction"),
        title="Political Speech Classifier",
        description="Enter text to classify using various classifiers."
    )
    iface.launch()

if __name__ == '__main__':
    main()