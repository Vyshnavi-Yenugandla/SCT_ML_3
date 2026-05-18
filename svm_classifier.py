import os
import cv2
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report

# Dataset path
dataset_path = "dataset"

categories = ['cats', 'dogs']

data = []
labels = []

IMG_SIZE = 64

# Load and preprocess images
for category in categories:

    folder_path = os.path.join(dataset_path, category)

    label = categories.index(category)

    for img_name in os.listdir(folder_path):

        img_path = os.path.join(folder_path, img_name)

        try:

            img = cv2.imread(img_path)

            # Convert to grayscale
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Resize image
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

            # Flatten image
            img = img.flatten()

            data.append(img)
            labels.append(label)

        except Exception as e:
            print("Error loading image:", img_path)

# Convert to NumPy arrays
X = np.array(data)
y = np.array(labels)

print("Dataset Loaded")
print("X Shape:", X.shape)
print("Y Shape:", y.shape)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train model
svm_model = LinearSVC()

print("Training Started...")

svm_model.fit(X_train, y_train)

print("Training Completed")

# Predictions
y_pred = svm_model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

print("\nClassification Report:\n")

print(classification_report(y_test, y_pred))

# Save model
joblib.dump(svm_model, "cats_dogs_model.pkl")

print("Model Saved Successfully")

# Display prediction
categories = ['Cat', 'Dog']

index = 0

sample = X_test[index].reshape(64, 64)

prediction = svm_model.predict([X_test[index]])

plt.imshow(sample, cmap='gray')

plt.title("Prediction: " + categories[prediction[0]])

plt.axis('off')

plt.show()