import os
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers, models, callbacks
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
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

    # Save the datasets as .npy files
    np.save("X_train.npy", X_train)
    np.save("X_val.npy", X_val)
    np.save("X_test.npy", X_test)
    np.save("Y_train.npy", y_train)
    np.save("Y_val.npy", y_val)
    np.save("Y_test.npy", y_test)

    print("Size of Training Dataset:", len(X_train))
    print("Size of Validation Dataset:", len(X_val))
    print("Size of Test Dataset:", len(X_test))

    return X_train, X_val, y_train, y_val

def predict_image():
    pass

def augment_data():
    pass

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

def PlotMetrics(trainLosses, trainAccs, valLosses, valAccs, epochs):
    """
        Given training & validations losses, plot their losses over all epochs.
    """
    epoch_axis = np.arange(0, epochs, 1)
    fig, (ax1, ax2) = plt.subplots(figsize=(8, 12), nrows=1, ncols=2)
    fig.tight_layout()

    # Loss plot
    ax1.plot(epoch_axis, trainLosses, c="lightblue", alpha=0.5, label="Training Loss")
    ax1.plot(epoch_axis, valLosses, c="orange", alpha=0.5, label="Validation Loss")

    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    loss_title_string = f"Training and Validation Cross-Entropy Loss after {epochs} epochs"
    ax1.set_title(loss_title_string)
    ax1.legend()

    # Accuracy Plot
    ax2.plot(epoch_axis, trainAccs, c="lightblue", alpha=0.5, label="Training Accuracy")
    ax2.plot(epoch_axis, valAccs, c="orange", alpha=0.5, label="Validation Accuracy")

    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Accuracy")
    acc_title_string = f"Training and Validation Accuracy after {epochs} epochs"
    ax2.set_title(acc_title_string)
    ax2.legend()

    plt.show()

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
    PlotMetrics(history.history['loss'], history.history['accuracy'], history.history['val_loss'], history.history['val_accuracy'], epochs)

def save_model(model, filename='chest_xray_model.pkl'):
    with open(filename, 'wb') as file:
        pickle.dump(model, file)
    print(f"Model saved as {filename}")

def main():
    directory_path = "Chest X_Ray Dataset"
    directories = ["COVID19", "NORMAL", "PNEUMONIA", "TURBERCULOSIS"]
    IMAGE_SIZE = 224
    test_valid_size = 0.3

    X_train, X_val, y_train, y_val = preprocess_data(directory_path, directories, IMAGE_SIZE, test_valid_size)

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

if __name__ == "__main__":
    main()
