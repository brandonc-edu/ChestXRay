import tensorflow as tf
from sklearn.metrics import recall_score, precision_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import pickle
import numpy as np
import matplotlib.pyplot as plt

def evaluate_model(model, X, y, set_name='Test', labels=[0, 1, 2, 3], classes=["Covid19", "Normal", "Pneumonia", "Turberculosis"]):
    """
        Evaluates final model performance on given X & y set using a variety of metrics & graphical analyses.

        Metrics used:
        - Accuracy
        - Recall
        - Precision
        - F1-Score
        
        Graphical Analyses:
        - ROC Curve (TO BE IMPLEMENTED)
        - Confusion Matrix (TO BE FULLY IMPLEMENTED)
    """
    loss, accuracy = model.evaluate(X, y)
    print(f"{set_name} Accuracy: {accuracy * 100:.2f}%")
    print(f"{set_name} Loss: {loss:.4f}")

    # Get predictions
    y_preds = np.argmax(model.predict(X), axis=1)
    y_true = np.argmax(y, axis=1)

    # Calculate metrics
    precision = precision_score(y_true, y_preds, average='weighted', labels=labels) * 100
    recall = recall_score(y_true, y_preds, average='weighted', labels=labels) * 100
    f1 = f1_score(y_true, y_preds, average='weighted', labels=labels) * 100
    conf_matrix = confusion_matrix(y_true, y_preds)

    # Print metrics
    print(f"Model {set_name} Precision: {precision:.2f}%")
    print(f"Model {set_name} Recall: {recall:.2f}%")
    print(f"Model {set_name} F1-Score: {f1:.2f}%")
    ## Below two lines may be phased out once plot_confusion_matrix() is finished.
    print(f"Confusion Matrix for {set_name} Set predictions:") 
    print(conf_matrix)
    plot_confusion_matrix(y_true, y_preds, classes)

    # Plot ROC Curves

def plot_confusion_matrix(y_true, y_preds, classes=None):
    """
        Plots the confusion matrix given true values & predictions.
    """
    # Plots a simple confusion matrix using the true values, predicted values, and (if provided) class labels.
    ConfusionMatrixDisplay.from_predictions(y_true, y_preds, display_labels=classes)
    plt.show()

def load_saved_model(filename='chest_xray_model.pkl'):
    with open(filename, 'rb') as file:
        model = pickle.load(file)
    print(f"Model loaded from {filename}")
    return model

def main():
    # Load Test Data
    X_test, y_test = np.load('X_test.npy'), np.load('Y_test.npy')

    # Load the saved model
    loaded_model = load_saved_model('chest_xray_model.pkl')

    # Evaluate the loaded model on the test set
    evaluate_model(loaded_model, X_test, y_test)

if __name__ == "__main__":
    main()