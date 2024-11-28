import os
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

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

def build_model():
    pass

def train_model():
    pass

def evaluate_model():
    pass

def save_model():
    pass

def load_model():
    pass

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

    return

if __name__ == "__main__":
    main()