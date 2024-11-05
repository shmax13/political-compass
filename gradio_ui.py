import joblib
import os
import itertools
from collections import Counter
import numpy as np
import gradio as gr
import matplotlib.pyplot as plt
from util.preprocessing import preprocess_text

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
    """Make predictions for classification and regression with calibration."""
    if not input_text.strip():  # Handle empty input with calibrated origin (0, 0)
        final_classification_prediction = "No Prediction"
        final_x_prediction, final_y_prediction = 0.0, 0.0
        plot_image_path = plot_coordinates(final_x_prediction, final_y_prediction)
        return final_classification_prediction, (final_x_prediction, final_y_prediction), plot_image_path

    # Proceed with classification and regression predictions
    classifiers = ['Logistic_Regression', 'Random_Forest_Classifier', 'Support_Vector_Classifier']
    regressors = ['Linear_Regression_(x)', 'Linear_Regression_(y)', 
                  'Random_Forest_Regressor_(x)', 'Random_Forest_Regressor_(y)',
                  'Support_Vector_Regressor_(x)', 'Support_Vector_Regressor_(y)']
    vectorizers = ['TfidfExtractor', 'BagOfWordsExtractor', 'NgramsExtractor', 'word2vecExtractor', 'BERTExtractor']    


    all_predictions = []
    all_regression_predictions = {'x': [], 'y': []}

    processed_text = preprocess_text(input_text)
    # Classify the input text
    for extractor_name, classifier_name in itertools.product(vectorizers, classifiers):
        model_path = f'classifiers/{extractor_name}_{classifier_name}.pkl'
        vectorizer_path = f'vectorizers/{extractor_name}_vectorizer.pkl'

        if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
            continue

        model = load_model(model_path)
        vectorizer = load_vectorizer(vectorizer_path)

        prediction = classify_input(model, vectorizer, processed_text)
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
        prediction = predict_regression(regressor, vectorizer, processed_text)
        if 'x' in regressor_name:
            all_regression_predictions['x'].append(prediction)
        else:
            all_regression_predictions['y'].append(prediction)

    # Calculate the majority vote for classification
    final_classification_prediction = Counter(all_predictions).most_common(1)[0][0] if all_predictions else "No Prediction"

    # Calculate the average of the x and y regression predictions
    final_x_prediction = np.median(all_regression_predictions['x']) if all_regression_predictions['x'] else 0.0
    final_y_prediction = np.median(all_regression_predictions['y']) if all_regression_predictions['y'] else 0.0

    # Format the regression predictions to 3 decimals
    final_x_prediction = f"{final_x_prediction:.3f}"
    final_y_prediction = f"{final_y_prediction:.3f}"

    # Plot the calibrated coordinates and save the plot image
    plot_image_path = plot_coordinates(final_x_prediction, final_y_prediction)

    return final_classification_prediction, (final_x_prediction, final_y_prediction), plot_image_path


def plot_coordinates(x,y): 
    """plot the x and y coordinates as point in "D coordinate space - compass"""

      # Convert x and y to float for plotting
    x = float(x)
    y = float(y)

    # Create plot
    plt.figure(figsize=(6,6))
    

    # Add political compass labels on axes
    plt.text(1.02, 0.5, 'Right', transform=plt.gca().transAxes, ha='center', fontsize=12)
    plt.text(-0.08, 0.5, 'Left', transform=plt.gca().transAxes, ha='center', fontsize=12)
    plt.text(0.5, 1.02, 'Authoritarian', transform=plt.gca().transAxes, va='center', fontsize=12)
    plt.text(0.5, -0.08, 'Libertarian', transform=plt.gca().transAxes, va='center', fontsize=12)

    # Set axis limits to zoom in on the scale
    plt.xlim(-10.5, 10.5)
    plt.ylim(-10.5, 10.5)
    plt.axis('equal')

    # plt.title('Political compass plot')
    plt.grid(True)

    # Add colors to quadrants 
    colors = ['red', 'blue', 'green', 'purple']

    # Define the limits of the plot
    x_min, x_max = -10.5, 10.5
    y_min, y_max = -10.5, 10.5

    # Calculate the quadrant boundaries
    x1, x2 = (x_min, x_max)
    y1, y2 = (y_min, y_max)

    # Plot the lines and fill the quadrants
    plt.plot([0, 0], [y1, y2], color='black', linewidth=1)
    plt.plot([x1, x2], [0, 0], color='black', linewidth=1)

    # Fill the quadrants
    plt.fill_between([x_min, 0], [y_max, y_max], color=colors[0])  # top left (red)
    plt.fill_between([0, x_max], [y_max, y_max], color=colors[1])  # top right (blue)
    plt.fill_between([x_min, 0], [y_min, y_min], color=colors[2])  # bottom left (green)
    plt.fill_between([0, x_max], [y_min, y_min], color=colors[3])  # bottom right (purple)

    #plot point
    plt.scatter(x,y, color='white', marker='x', linewidths=2)
    plt.text(x + 0.15, y + 0.15, f'({x}, {y})', fontsize=10, ha='left', color='white') #+ offset to prevent label overlap 
    
    #save plot as png - to show in ui 
    plt_image_path = 'compass_plot.png'
    plt.savefig(plt_image_path)

    return plt_image_path


def main():
    # Create a Gradio interface
    iface = gr.Interface(
        fn=make_prediction, 
        inputs=gr.Textbox(label="Input Text"), 
        outputs=[
            gr.Label(label="Classification Prediction"), 
            gr.Textbox(label="Regression Coordinates (x, y)"),
            gr.Image(label="Political Compass Plot")
        ],
        title="Political Speech Classifier and Regression",
        description="Enter text to classify using various classifiers and predict political coordinates."
    )
    iface.launch()

if __name__ == '__main__':
    main()