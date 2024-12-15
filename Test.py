import tensorflow as tf
from sklearn.metrics import recall_score, precision_score, f1_score, confusion_matrix, ConfusionMatrixDisplay, auc, roc_curve, RocCurveDisplay
from sklearn.manifold import TSNE
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def balanced_accuracy(y_true, y_preds):
    """
        Given true values and predictions, finds the balanced accuracy of the predictions.

        Credit: https://arxiv.org/pdf/2008.05756
    """
    conf_mat = confusion_matrix(y_true, y_preds)
    class_counts = np.sum(conf_mat, axis=1)
    # cl_weights = class_counts / np.sum(conf_mat)

    K = len(conf_mat)
    numerator = 0
    for k in range(K):
        numerator += conf_mat[k, k] / class_counts[k]
    return numerator / K
    

def evaluate_model(model, X, y, set_name='Test', labels=[0, 1, 2, 3], classes=["Covid19", "Normal", "Pneumonia", "Turberculosis"]):
    """
        Evaluates final model performance on given X & y set using a variety of metrics & graphical analyses.

        Metrics used:
        - Accuracy
        - Balanced Accuracy (among all classes)
        - Recall
        - Precision
        - F1-Score
        
        Graphical Analyses:
        - ROC Curve
        - Confusion Matrix
        - 2D (t-SNE) Kernel Density Plot (see: https://www.oranlooney.com/post/viz-tsne/)
        - Hit-or-Miss Plot
    """
    loss, accuracy = model.evaluate(X, y)
    print(f"{set_name} Accuracy: {accuracy * 100:.2f}%")
    print(f"{set_name} Loss: {loss:.4f}")

    # Get predictions
    y_logits = model.predict(X)
    y_preds = np.argmax(y_logits, axis=1)
    y_true = np.argmax(y, axis=1)

    # Calculate metrics
    bal_acc = balanced_accuracy(y_true, y_preds) * 100
    precision = precision_score(y_true, y_preds, average='weighted', labels=labels) * 100
    recall = recall_score(y_true, y_preds, average='weighted', labels=labels) * 100
    f1 = f1_score(y_true, y_preds, average='weighted', labels=labels) * 100
    conf_matrix = confusion_matrix(y_true, y_preds)

    # Print metrics
    print(f"Model {set_name} Balanced Accuracy: {bal_acc:.2f}%")
    print(f"Model {set_name} Precision: {precision:.2f}%")
    print(f"Model {set_name} Recall: {recall:.2f}%")
    print(f"Model {set_name} F1-Score: {f1:.2f}%")
    print(f"Confusion Matrix for {set_name} Set predictions:") 
    print(conf_matrix)
    plot_confusion_matrix(y_true, y_preds, set_name, classes)

    # Plot ROC Curves
    PlotROCCurve(y, y_logits, classes)

    # Plot 2D Kernel Density Plot using t-SNE
    plot_KDE_hit_miss(X, y_true, y_preds, classes)

    # if os.path.isfile('X_tsne_2D.npy'):
    #     X_reduced = np.load('X_tsne_2D.npy')
    # else:
    #     X_reduced = TSNE(random_state=42).fit_transform(X.reshape(len(X), -1))
    #     np.save('X_tsne_2D.npy', X_reduced)


def PlotROCCurve(y_true_raw, y_preds_proba, classes):
    """
        Given the oneshot true values and prediction probabilities of the model, generate several ROC Curves.
    """
    # Micro-averaged ROC Curve
    fpr_micro, tpr_micro, _ = roc_curve(y_true_raw.ravel(), y_preds_proba.ravel())
    roc_auc_micro = auc(fpr_micro, tpr_micro)

    fig, ax = plt.subplots(figsize=(12, 8))
    fig.tight_layout()

    plt.plot(fpr_micro, tpr_micro, label=f"Micro-average ROC Curve with AUC = {roc_auc_micro:.2f}", linestyle='--')

    # One versus rest class ROC Curves
    for i in range(len(classes)):
        class_label = classes[i]
        RocCurveDisplay.from_predictions(
            y_true_raw[:, i],
            y_preds_proba[:, i],
            name=f"ROC Curve ({class_label} vs. Rest)",
            ax=ax,
            plot_chance_level=(i == len(classes) - 1),
        )
    ax.set_xlabel("False Positive Rate (FPR)")
    ax.set_ylabel("True Positive Rate (TPR)")
    ax.set_title("Micro-average + One vs. Rest ROC Curves on all Classes")

    plt.legend()
    plt.show()


def plot_confusion_matrix(y_true, y_preds, set_name='Test', classes=None):
    """
        Plots the confusion matrix given true values & predictions.
    """
    # Plots a simple confusion matrix using the true values, predicted values, and (if provided) class labels.
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.tight_layout()
    ConfusionMatrixDisplay.from_predictions(y_true, y_preds, display_labels=classes, ax=ax)
    ax.set_title(f"Confusion Matrix on X-Ray {set_name} Predictions")
    plt.show()


def plot_KDE_hit_miss(X, y_true, y_preds, classes):
    """
        Given X matrix, true values, and predictions, plot a 2D (t-SNE) Kernel Density plot & hit-or-miss plot.
    """
    # Distill X down to 2D using t-SNE
    X_reduced = TSNE(random_state=42).fit_transform(X.reshape(len(X), -1))

    fig, (ax1, ax2) = plt.subplots(figsize=(8, 12), nrows=1, ncols=2)
    fig.tight_layout()

    # Plot KDE, splitting them by their true values
    kde_data = pd.DataFrame({
        "X" : X_reduced[:, 0],
        "Y" : X_reduced[:, 1],
        "class" : np.take(classes, y_true)
    })
    sns.kdeplot(kde_data, x="X", y="Y", hue="class", fill=True, ax=ax1)
    # ax1.legend(classes)
    ax1.set_title("2D (t-SNE) Kernel Density Plot")

    ## Plot accompanying hit-or-miss plot to see if the densities align with (relatively) heavy misclassification areas.
    hit_or_miss = (y_true == y_preds).astype(int)
    hits = np.asarray(hit_or_miss == 1).nonzero()
    misses = np.asarray(hit_or_miss == 0).nonzero()

    ax2.scatter(np.take(X_reduced[:, 0], hits), np.take(X_reduced[:, 1], hits), c='blue', label="Correct Predictions")
    ax2.scatter(np.take(X_reduced[:, 0], misses), np.take(X_reduced[:, 1], misses), c='red', label="Incorrect Predictions")

    ax2.legend()
    ax2.set_title("Hit-or-Miss Plot (t-SNE)")
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