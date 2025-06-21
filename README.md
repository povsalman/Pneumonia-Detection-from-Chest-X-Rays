# Pneumonia Detection from Chest X-Rays
A deep learning project to detect pneumonia from chest X-ray images using LSTM (RNN) and MobileNet (CNN) models, based on a Kaggle dataset and inspired by a research paper.

# Project Overview

Objective: Automate pneumonia detection using X-ray images with high accuracy (>95%).

Dataset: Kaggle "Chest X-Ray Images (Pneumonia)" (5856 images: NORMAL, PNEUMONIA).

Models: LSTM (custom, per research paper) and MobileNet (transfer learning).

Features: Data preprocessing, model training, evaluation, Grad-CAM visualization.

# Dataset

Source: Kaggle Chest X-Ray Pneumonia.

Split:

1) Training: ~4694 images (80% of original train, NORMAL: ~1207, PNEUMONIA: ~3487).
2) Validation: ~522 images (20% of original train, NORMAL: ~134, PNEUMONIA: ~387).
3) Test: 624 images (NORMAL: 234, PNEUMONIA: 390).


# Preprocessing:
Resize images to 150x150, normalize to [0, 1].

Reshape for LSTM: (150, 450) from 150x150x3.

Class weights for imbalance (NORMAL: ~2.89, PNEUMONIA: ~0.35).


# Models

1} LSTM :

  * 3 LSTM layers (250, 120, 64 units, dropout=0.05, recurrent_dropout=0.20).

  * Dense layers: 250, 120, 64, 28 (ReLU), 2 (softmax).

  * Optimizer: Adam, Loss: Categorical cross-entropy.


2} MobileNet:
   
  * Pre-trained on ImageNet, frozen base layers.

  * GlobalAveragePooling2D, Dense(128, ReLU), Dense(2, softmax).

  * Optimizer: Adam, Loss: Categorical cross-entropy.


Training: 10 epochs, batch size=32, save best model via ModelCheckpoint.

# Evaluation

Metrics: Accuracy, precision, recall, F1-score, confusion matrix.

Visualization: Training/validation accuracy/loss curves.

Interpretability: Grad-CAM heatmaps for MobileNet, highlighting lung regions.

# Requirements

Python 3.11

Libraries: tensorflow, opencv-python, scikit-learn, matplotlib, seaborn, kagglehub

# Usage

1) Clone the repository: git clone https://github.com/povsalman/Pneumonia-Detection-from-Chest-X-Rays.git
2) Download dataset: Run the first cell in Model_Code.ipynb to fetch and split data.
3) Preprocess data: Run preprocessing cell in Model_Code.ipynb.
4) Train models: Run the Model training cell in Model_Code.ipynb or use the already provided model directly.
5) Evaluate: Run the Evaluate cell in Model_Code.ipynb for metrics and plots.
6) Predict: Run main cell for single-image prediction.

# Results

LSTM: ~80.2% accuracy, actual results depend on split and choice of hyperparameters.

MobileNet: ~90% accuracy, often outperforms LSTM in efficiency.

# Challenges

Class imbalance addressed with weights.

LSTM preprocessing is computationally intensive.

# Acknowledgments

Inspired by: S. Hossain, Rafeed Rahman, Pneumonia Detection by Analyzing Xray Images Using MobileNET ResNET Architecture and Long Short Term Memory(2020).

Dataset: Kaggle by Paul Timothy Mooney. ( https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia )

