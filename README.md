🩺 Detection of COVID-19, Pneumonia, and Tuberculosis from Chest X-Ray Images Using CNN
📘 Overview

This project focuses on building a Convolutional Neural Network (CNN) model to automatically classify chest X-ray images into three major disease categories — COVID-19, Pneumonia, and Tuberculosis — along with Normal cases.
The goal is to assist healthcare professionals by providing a fast, reliable, and automated diagnostic tool based on medical imaging.

Developed as part of an academic assignment on Digital Image Processing (DIP).

🎯 Objectives

Build a CNN-based image classification model for detecting COVID-19, Pneumonia, and Tuberculosis.

Apply data preprocessing and augmentation to improve model generalization.

Evaluate model performance using metrics like accuracy, precision, recall, and F1-score.

Visualize training results using confusion matrices, loss curves, and accuracy graphs.

🧩 Dataset

Dataset Name: Chest X-Ray Dataset (4 Categories)

Source: Kaggle - pritpal2873/chest-x-ray-dataset-4-categories

Categories:

COVID-19

Pneumonia

Tuberculosis

Normal

The dataset includes labeled X-ray images used to train, validate, and test the CNN model.

⚙️ Methodology
1. Data Preparation

Downloaded dataset using KaggleHub

Split into train (70%), validation (15%), and test (15%) sets

Resized and normalized all X-ray images

2. Model Architecture

The CNN architecture includes:

Multiple convolutional layers with ReLU activation

MaxPooling for feature reduction

Dropout layers to prevent overfitting

Fully connected (Dense) layers for classification

Softmax output layer for multi-class prediction

3. Training

Loss Function: categorical_crossentropy

Optimizer: Adam

Metrics: accuracy

Early stopping applied to prevent overfitting

4. Evaluation

Accuracy, Precision, Recall, F1-score

Confusion Matrix

Visual plots for training and validation metrics

📊 Results
Metric	Value (Example)
Training Accuracy	~95%
Validation Accuracy	~90%
Test Accuracy	~88%
Loss Curve	Smooth convergence observed

✅ The CNN model demonstrated strong performance in differentiating between normal and diseased X-ray images.
Visualization of the confusion matrix showed high classification confidence for Pneumonia and COVID-19 categories.

🧰 Tools & Libraries

Python 🐍

TensorFlow / Keras

NumPy, Pandas, Matplotlib, Seaborn

Scikit-learn

KaggleHub

OpenCV / Pillow for image preprocessing

🚀 How to Run
# 1. Clone this repository
git clone https://github.com/FSweety/ChestDiseaseDetection-CNN.git
cd ChestDiseaseDetection-CNN

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the notebook
jupyter notebook "DIP_Theory_Assignment(ChestDiseaseDetection).ipynb"

🩻 Model Workflow

Load dataset from Kaggle

Split into train/val/test sets

Preprocess and augment images

Train CNN model

Evaluate model and visualize results

💡 Future Improvements

Implement transfer learning using pre-trained models (ResNet, VGG16, EfficientNet).

Expand dataset diversity across different demographics.

Build a web-based interface for real-time image upload and disease prediction.

Integrate explainable AI (Grad-CAM) to visualize which parts of X-ray images influence predictions.
