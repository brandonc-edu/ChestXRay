import os
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.models import load_model
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import recall_score, precision_score, accuracy_score, f1_score, confusion_matrix
import pickle

# Preprocess the data to be used in the model
def preprocess_data(path, directories, IMAGE_SIZE, test_val_size):
    # Retrieve directory of the dataset
    path = os.path.abspath(path) + "/"

    # For plotting purposes
    samples = [] # Store a sample image from each category
    count_images = [] # Count number of data samples in each category


    xray_images = [] # Store all images
    target = [] # Store all target labels

    for i, directory in enumerate(directories):
        images = os.listdir(path + directory) # Get all images for each category
        sample_image = None # Store 1 sample image for each categor

        for image_name in images:
            # Convert image to grayscale, resize, and store image with label
            image = cv2.imread(path + directory + "/" + image_name, cv2.IMREAD_GRAYSCALE)
            resized_image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
            xray_images.append(resized_image)
            target.append(i)

            # Only need the first sample image for each category
            if sample_image is None:
                sample_image = resized_image
        
        # Plotting purposes
        samples.append(sample_image)
        count_images.append(len(images))
    
    # Plot class distrubution
    plot_class_distribution(directories, count_images)

    # Plot samples images
    plot_sample_images(directories, samples)
    
    # Convert data into numpy arrays
    xray_images = np.array(xray_images)
    target = np.array(target)

    # Reshape and normalize
    xray_images = xray_images.reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1)
    xray_images = xray_images / 255.0

    # One-hot encode labels
    target = to_categorical(target, num_classes=len(directories))

    # Split data into train, validate, and test sets
    X_train, X_test_valid, y_train, y_test_valid = train_test_split(xray_images, target, test_size = test_val_size, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_test_valid, y_test_valid, test_size = 0.5, random_state=42)

    print("Size of Training Dataset:", len(X_train))
    print("Size of Validation Dataset:", len(X_val))
    print("Size of Test Dataset:", len(X_test))

    return X_train, X_val, X_test, y_train, y_val, y_test

# Plot class distribution
def plot_class_distribution(categories, num_images_per_category):
    plt.figure(figsize=(10, 6))
    plt.bar(categories, num_images_per_category, color='blue')
    plt.xlabel('Categories')
    plt.ylabel('Number of Samples')
    plt.title('Class Distribution (# of Samples)')
    plt.show()

# Plot sample images
def plot_sample_images(categories, samples):
    plt.figure(figsize=(12, 8))
    for i, category in enumerate(categories):
        plt.subplot(1, len(categories), i + 1)
        plt.imshow(samples[i], cmap='gray')
        plt.title(category)
        plt.axis('off')
    plt.suptitle('Grayscaled and Resized Chest Xrays from Each Category')
    plt.show()

def build_model(input_shape, num_classes):
    """
    Build a CNN model.
    """
    model = models.Sequential()

    # Convolutional and Pooling Layers
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    # Flatten and Fully Connected Layers
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes, activation='softmax'))

    return model

def train_model(model, X_train, y_train, X_val, y_val, batch_size, epochs, class_weights=None):
    """
    Train the CNN model with class weights.
    """
    model.compile(
        optimizer='adam',  # Optimizer
        loss='categorical_crossentropy',  # Loss function
        metrics=['accuracy']  # Evaluation metric
    )

    early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=[early_stop],
        class_weight=class_weights 
    )

    return model, history

def calculate_metrics(history):
    """
    Calculate performance metrics from training history.

    TO BE IMPLEMENTED: Plot of losses & accuracy over training
    """
    epochs = len(history.history['accuracy'])

    train_accuracy = history.history['accuracy'][-1]
    val_accuracy = history.history['val_accuracy'][-1]
    train_loss = history.history['loss'][-1]
    val_loss = history.history['val_loss'][-1]

    print(f"Final Training Accuracy: {train_accuracy:.4f}")
    print(f"Final Validation Accuracy: {val_accuracy:.4f}")
    print(f"Final Training Loss: {train_loss:.4f}")
    print(f"Final Validation Loss: {val_loss:.4f}")

    # Plot the history of loss + accuracy over training.

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
        - Confusion Matrix (TO BE IMPLEMENTED)
    """
    loss, accuracy = model.evaluate(X, y)
    print(f"{set_name} Accuracy: {accuracy:.4f}")
    print(f"{set_name} Loss: {loss:.4f}")

    # Get predictions
    y_preds = model.predict(X).numpy()
    y_true = y.numpy()

    # Calculate metrics
    precision = precision_score(y_true, y_preds)
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

    # Plot ROC Curve



def save_model(model, filename='chest_xray_model.pkl'):
    with open(filename, 'wb') as file:
        pickle.dump(model, file)
    print(f"Model saved as {filename}")

def load_saved_model(filename='chest_xray_model.pkl'):
    with open(filename, 'rb') as file:
        model = pickle.load(file)
    print(f"Model loaded from {filename}")
    return model

def predict_image():
    pass

def augment_data():
    pass

def plot_confusion_matrix(conf_mat):
    """
        Plots the confusion matrix given confusion matrix.
    """
    pass

def main():
    directory_path = "data"
    directories = ["COVID19", "NORMAL", "PNEUMONIA", "TURBERCULOSIS"]
    IMAGE_SIZE = 224
    test_valid_size = 0.3

    X_train, X_val, X_test, y_train, y_val, y_test = preprocess_data(directory_path, directories, IMAGE_SIZE, test_valid_size)

    # Model Parameters
    input_shape = (IMAGE_SIZE, IMAGE_SIZE, 1)  # Grayscale images
    num_classes = len(directories)
    batch_size = 32
    epochs = 25

    # Compute class weights based on training labels
    class_weights = compute_class_weight(
        class_weight='balanced', 
        classes=np.unique(y_train.argmax(axis=1)), 
        y=y_train.argmax(axis=1)
    )
    class_weights = dict(enumerate(class_weights))  # Convert to dictionary
    
    # Build and Train Model
    model = build_model(input_shape, num_classes)
    model, history = train_model(
        model, X_train, y_train, X_val, y_val, batch_size, epochs, class_weights=class_weights
    )

    # Evaluate and Report Metrics
    calculate_metrics(history)
    
    # Save the model for further evaluation
    save_model(model)
    
    # Load the saved model
    loaded_model = load_saved_model('chest_xray_model.pkl')

    # Evaluate the loaded model on the test set
    evaluate_model(loaded_model, X_test, y_test)
    
    return

if __name__ == "__main__":
    main()