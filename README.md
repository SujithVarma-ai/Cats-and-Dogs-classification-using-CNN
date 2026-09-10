# 🐱🐶 Cats & Dogs Image Classifier using CNN

A deep learning project that classifies images as **Cat** or **Dog** using a custom Convolutional Neural Network (CNN) built with **TensorFlow and Keras**.

## 📌 Project Overview

This project demonstrates an end-to-end image classification pipeline using a Convolutional Neural Network.

The project:

- Loads the Cats vs Dogs dataset using TensorFlow Datasets
- Extracts and organizes the dataset into class-specific directories
- Preprocesses images by resizing and normalizing them
- Applies data augmentation to improve generalization
- Builds a custom CNN using Keras
- Uses Convolution, Max Pooling, Batch Normalization, Dropout, Flatten, and Dense layers
- Trains the model for binary classification
- Visualizes training and validation performance
- Saves the trained model
- Loads the saved model for later predictions

## 📂 Dataset

The project uses the **Cats vs Dogs** dataset through `tensorflow_datasets`.

```python
dataset, info = tfds.load(
    'cats_vs_dogs',
    with_info=True,
    as_supervised=True
)
