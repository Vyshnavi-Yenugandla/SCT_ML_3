# Cats vs Dogs Classification using SVM

## Description

This project implements a machine learning model using Support Vector Machine (SVM) to classify images of cats and dogs. The images are preprocessed using OpenCV and trained using Scikit-learn.

The model predicts whether the given image belongs to a cat or a dog class.

---

## Technologies Used

* Python
* OpenCV
* NumPy
* Scikit-learn
* Matplotlib
* Joblib

---

## Dataset

Dataset used from Kaggle:

[Dogs vs Cats Dataset](https://www.kaggle.com/datasets/salader/dogs-vs-cats?utm_source=chatgpt.com)

---

## Features

* Image preprocessing
* Image resizing
* Grayscale conversion
* Feature extraction
* SVM model training
* Accuracy evaluation
* Prediction on test images
* Model saving using Joblib

---

## Project Structure

```text id="fhrp55"
cats-dogs-svm/
│
├── dataset/
│   ├── cats/
│   └── dogs/
│
├── svm_classifier.ipynb
├── cats_dogs_model.pkl
├── README.md
└── outputs/
```

---

## Installation

Install required libraries:

```bash id="2rg2jq"
pip install numpy opencv-python matplotlib scikit-learn joblib
```

---

## How to Run

### 1. Open Jupyter Notebook

```bash id="u8ks0k"
jupyter notebook
```

### 2. Run the notebook

Open:

```text id="p2mk8m"
svm_classifier.ipynb
```

Run all cells sequentially.

---

## Model Workflow

1. Load images from dataset
2. Resize images
3. Convert images to grayscale
4. Flatten images into feature vectors
5. Split dataset into train and test sets
6. Train SVM classifier
7. Predict image classes
8. Evaluate model accuracy

---

## Output Example

```text id="wq4tn0"
Prediction: Cat
Accuracy: 85%
```

---

## Save Trained Model

```python id="pm5g1r"
import joblib

joblib.dump(svm_model, "cats_dogs_model.pkl")
```

---

## Load Saved Model

```python id="g00z9v"
loaded_model = joblib.load("cats_dogs_model.pkl")
```


## Future Improvements

* Use CNN/Deep Learning models
* Improve accuracy using HOG features
* Real-time webcam prediction
* Data augmentation
* Hyperparameter tuning
## Conclusion

This project demonstrates how machine learning and computer vision techniques can be used to classify images using Support Vector Machine (SVM). It provides hands-on experience in image preprocessing, model training, and prediction using Python libraries.
