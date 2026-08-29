# MNIST Handwritten Digit Recognition

## Project Overview

A CNN-based handwritten digit recognition system that classifies handwritten digits from 0 to 9 using the MNIST dataset.

The project includes model training, evaluation, single-image prediction, and a Streamlit web application for interactive digit recognition.

## Objectives

- Build a Convolutional Neural Network (CNN) for handwritten digit classification.
- Apply image preprocessing and normalization.
- Use Batch Normalization, Dropout, and Early Stopping.
- Evaluate the model using accuracy, classification report, and confusion matrix.
- Predict digits from individual images.
- Build an interactive Streamlit application.
- Display prediction confidence and class probabilities.

## Dataset

The project uses the MNIST handwritten digit dataset provided through TensorFlow.

The dataset contains:

- 60,000 training images
- 10,000 testing images
- Image size: 28 × 28 pixels
- Image type: Grayscale
- Number of classes: 10
- Classes: 0–9

## CNN Architecture

```text
Input Image (28 × 28 × 1)
        ↓
Rescaling
        ↓
Conv2D (32 filters)
        ↓
Batch Normalization
        ↓
Max Pooling
        ↓
Conv2D (64 filters)
        ↓
Batch Normalization
        ↓
Max Pooling
        ↓
Conv2D (128 filters)
        ↓
Batch Normalization
        ↓
Max Pooling
        ↓
Flatten
        ↓
Dense (256)
        ↓
Dropout
        ↓
Dense (128)
        ↓
Dropout
        ↓
Dense (10)
        ↓
Softmax
        ↓
Digit Prediction