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
```

### Classes

The model performs binary classification between two classes:

| Label | Class |
|------:|-------|
| 0 | Cat |
| 1 | Dog |

The class names are obtained from the dataset:

```python
class_names = info.features['label'].names
```

### Output

```bash
['cat','dog']
```

## 🔄 Data Preparation

The dataset images are organized into class-specific directories:

```text
cats_vs_dogs/
└── train/
    ├── cat/
    └── dog/
```

## 🖼️ Image Preprocessing

Images are resized to:

```bash
150 × 150 × 3
```

where:

150 × 150 represents the image height and width

3 represents the RGB color channels

## 📊 Training and Validation Data

The dataset is divided using an 80/20 training-validation split.

The notebook produces:

Training images   : 18,611
Validation images : 4,651
Classes           : 2
Batch size        : 32

## 🏗️ CNN Architecture

```bash
Input Image
150 × 150 × 3
        ↓
Conv2D
32 filters, 3×3
ReLU
        ↓
MaxPooling2D
2×2
        ↓
Batch Normalization
        ↓
Dropout
0.2
        ↓
Conv2D
64 filters, 3×3
ReLU
        ↓
MaxPooling2D
2×2
        ↓
Batch Normalization
        ↓
Dropout
0.2
        ↓
Conv2D
128 filters, 3×3
ReLU
        ↓
MaxPooling2D
2×2
        ↓
Batch Normalization
        ↓
Dropout
0.2
        ↓
Flatten
        ↓
Dropout
0.2
        ↓
Dense
512 neurons
ReLU
        ↓
Dense
1 neuron
Sigmoid
        ↓
Cat / Dog
```
