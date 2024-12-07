from tensorflow.keras.models import load_model
import tensorflow as tf
from sklearn.metrics import recall_score, precision_score, accuracy_score, f1_score, confusion_matrix
import pickle

def evaluate_model(model, X, y, set_name='Test'):
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
    print(f"{set_name} Accuracy: {accuracy:.4f}")
    print(f"{set_name} Loss: {loss:.4f}")

    # Get predictions
    y_preds = model.predict(X)
    y_true = y

    # Calculate metrics
    precision = precision_score(y_true, y_preds, labels=["Boo"])
    recall = recall_score(y_true, y_preds)
    f1 = f1_score(y_true, y_preds)
    conf_matrix = confusion_matrix(y_true, y_preds)

    # Print metrics
    print(f"Model {set_name} Precision: {precision:.4f}")
    print(f"Model {set_name} Recall: {recall:.4f}")
    print(f"Model {set_name} f1-score: {f1:.4f}")
    ## Below two lines will be phased out once plot_confusion_matrix() is finished.
    print(f"Confusion Matrix for {set_name} Set predictions:") 
    print(conf_matrix)
    # plot_confusion_matrix(conf_matrix)

    # Plot ROC Curves

def plot_confusion_matrix(conf_mat):
    """
        Plots the confusion matrix given confusion matrix.
    """
    pass

def load_saved_model(filename='chest_xray_model.pkl'):
    with open(filename, 'rb') as file:
        model = pickle.load(file)
    print(f"Model loaded from {filename}")
    return model

def main():
    # Load Test Data

    # Load the saved model
    loaded_model = load_saved_model('chest_xray_model.pkl')

    # Evaluate the loaded model on the test set
    evaluate_model(loaded_model, X_test, y_test)

if __name__ == "__main__":
    main()