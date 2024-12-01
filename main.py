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

def preprocess_data(data_path, categories, img_size, test_size, val_size, random_state):
    # Create lists to store data and labels
    data = []
    labels = []

    # Load the images and their respective labels
    print("Loading images...")
    for category in categories:
        path = os.path.join(data_path, category)
        class_num = categories.index(category)  # Assign class label
        for img in os.listdir(path):
            try:
                img_path = os.path.join(path, img)
                image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Load the image in grayscale
                image = cv2.resize(image, (img_size, img_size))  # Resize image to specified size
                data.append(image)
                labels.append(class_num)
            except Exception as e:
                print(f"Error loading image {img}: {e}")

    print("Image loading complete.")

    # Convert data and labels to numpy arrays
    data = np.array(data).reshape(-1, img_size, img_size, 1)
    data = data / 255.0  # Normalize pixel values to the range [0, 1]
    labels = np.array(labels)

    # One-hot encode the labels
    labels = to_categorical(labels, num_classes=len(categories))

    # Split dataset into training, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(data, labels, test_size=test_size, random_state=random_state)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=val_size, random_state=random_state)

    # Print dataset sizes
    print(f"Training set size: {len(X_train)}")
    print(f"Validation set size: {len(X_val)}")
    print(f"Test set size: {len(X_test)}")

    # Visualize label distribution
    label_counts = pd.Series(labels.argmax(axis=1)).value_counts(normalize=True) * 100
    plt.figure(figsize=(10, 6))
    plt.bar(categories, label_counts, color='skyblue')
    plt.xlabel('Class')
    plt.ylabel('Percentage')
    plt.title('Label Distribution (%)')
    plt.show()

    # Visualize a sample image from each class
    plt.figure(figsize=(12, 8))
    for i, category in enumerate(categories):
        path = os.path.join(data_path, category)
        img_path = os.path.join(path, os.listdir(path)[0])
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (img_size, img_size))
        plt.subplot(1, len(categories), i + 1)
        plt.imshow(image, cmap='gray')
        plt.title(category)
        plt.axis('off')
    plt.suptitle('Sample Grayscale Images from Each Class')
    plt.show()

    return X_train, X_val, X_test, y_train, y_val, y_test

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
    """
    train_accuracy = history.history['accuracy'][-1]
    val_accuracy = history.history['val_accuracy'][-1]
    train_loss = history.history['loss'][-1]
    val_loss = history.history['val_loss'][-1]

    print(f"Final Training Accuracy: {train_accuracy:.4f}")
    print(f"Final Validation Accuracy: {val_accuracy:.4f}")
    print(f"Final Training Loss: {train_loss:.4f}")
    print(f"Final Validation Loss: {val_loss:.4f}")

def evaluate_model(model, X_test, y_test):
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Loss: {test_loss:.4f}")

def save_model(model, filename='chest_xray_model.h5'):
    model.save(filename)
    print(f"Model saved as {filename}")

def load_saved_model(filename='chest_xray_model.h5'):
    model = load_model(filename)
    print(f"Model loaded from {filename}")
    return model

def predict_image():
    pass

def augment_data():
    pass

def plot_confusion_matrix():
    pass

def main():
    data_path="data"
    categories=["COVID19", "NORMAL", "PNEUMONIA", "TURBERCULOSIS"]
    img_size=224
    test_size=0.3
    val_size=0.5
    random_state=42

    # Preprocess the chest x-ray data
    X_train, X_val, X_test, y_train, y_val, y_test = preprocess_data(data_path, categories, img_size, test_size, val_size, random_state)

    # Model Parameters
    input_shape = (img_size, img_size, 1)  # Grayscale images
    num_classes = len(categories)
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
    loaded_model = load_saved_model('chest_xray_model.h5')

    # Evaluate the loaded model on the test set
    evaluate_model(loaded_model, X_test, y_test)
    
    return

if __name__ == "__main__":
    main()