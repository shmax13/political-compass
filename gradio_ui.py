import joblib
import os
import itertools
from collections import Counter
import numpy as np
import gradio as gr

# Caching for models and vectorizers to avoid reloading them every time
model_cache = {}
vectorizer_cache = {}

def load_model(model_path):
    """Load the trained model from a file and cache it."""
    if model_path in model_cache:
        return model_cache[model_path]
    
    model = joblib.load(model_path)
    model_cache[model_path] = model
    return model

def load_vectorizer(vectorizer_path):
    """Load the vectorizer and cache it."""
    if vectorizer_path in vectorizer_cache:
        return vectorizer_cache[vectorizer_path]
    
    vectorizer = joblib.load(vectorizer_path)
    vectorizer_cache[vectorizer_path] = vectorizer
    return vectorizer

def classify_input(model, vectorizer, input_text):
    """Classify the input text using the provided model."""
    features = vectorizer.transform([input_text])  
    prediction = model.predict(features)
    return prediction[0]

def predict_regression(regressor, vectorizer, input_text):
    """Predict the regression output (x or y coordinates) using the regressor."""
    features = vectorizer.transform([input_text])
    prediction = regressor.predict(features)
    return prediction[0]

def make_prediction(input_text):
    """Make predictions for classification and regression based on user input."""
    classifiers = ['Logistic_Regression', 'Random_Forest_Classifier', 'Support_Vector_Classifier']
    regressors = ['Linear_Regression_(x)', 'Linear_Regression_(y)', 
                  'Random_Forest_Regressor_(x)', 'Random_Forest_Regressor_(y)',
                  'Support_Vector_Regressor_(x)', 'Support_Vector_Regressor_(y)']
    vectorizers = ['TfidfExtractor', 'BagOfWordsExtractor', 'NgramsExtractor']

    all_predictions = []
    all_regression_predictions = {'x': [], 'y': []}

    # Classify the input text
    for extractor_name, classifier_name in itertools.product(vectorizers, classifiers):
        model_path = f'classifiers/{extractor_name}_{classifier_name}.pkl'
        vectorizer_path = f'vectorizers/{extractor_name}_vectorizer.pkl'

        if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
            continue

        model = load_model(model_path)
        vectorizer = load_vectorizer(vectorizer_path)

        prediction = classify_input(model, vectorizer, input_text)
        all_predictions.append(prediction)

    # Predict the x and y coordinates for regression
    for extractor_name, regressor_name in itertools.product(vectorizers, regressors):
        regressor_path = f'regressors/{extractor_name}_{regressor_name}.pkl'
        vectorizer_path = f'vectorizers/{extractor_name}_vectorizer.pkl'

        if not os.path.exists(regressor_path) or not os.path.exists(vectorizer_path):
            continue

        regressor = load_model(regressor_path)
        vectorizer = load_vectorizer(vectorizer_path)

        # Predict either the x or y coordinate
        prediction = predict_regression(regressor, vectorizer, input_text)
        if 'x' in regressor_name:
            all_regression_predictions['x'].append(prediction)
        else:
            all_regression_predictions['y'].append(prediction)

    # Calculate the majority vote for classification
    final_classification_prediction = Counter(all_predictions).most_common(1)[0][0] if all_predictions else "No Prediction"

    # Calculate the average of the x and y regression predictions
    final_x_prediction = np.mean(all_regression_predictions['x']) if all_regression_predictions['x'] else None
    final_y_prediction = np.mean(all_regression_predictions['y']) if all_regression_predictions['y'] else None

    # Format the regression predictions to 4 decimals 
    final_x_prediction = f"{final_x_prediction:.3f}" if final_x_prediction is not None else None
    final_y_prediction = f"{final_y_prediction:.3f}" if final_y_prediction is not None else None

    return final_classification_prediction, (final_x_prediction, final_y_prediction)

def main():
    # Create a Gradio interface
    iface = gr.Interface(
        fn=make_prediction, 
        inputs=gr.Textbox(label="Input Text"), 
        outputs=[
            gr.Label(label="Classification Prediction"), 
            gr.Textbox(label="Regression Coordinates (x, y)")
        ],
        title="Political Speech Classifier and Regression",
        description="Enter text to classify using various classifiers and predict political coordinates."
    )
    iface.launch()

if __name__ == '__main__':
    main()