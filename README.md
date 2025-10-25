# Handwritten Digit Classification Using CNN on the MNIST Dataset
A deep learning project implementing Convolutional Neural Networks (CNNs) to classify handwritten digits from the MNIST dataset. This project demonstrates the complete pipeline of building, training, and optimizing a CNN model for image classification with extensive experimentation on regularization techniques and data augmentation.
## Project Overview
This project builds a robust CNN architecture for recognizing handwritten digits (0-9) with high accuracy. It covers the entire deep learning workflow including data preprocessing, model construction, training with optimization techniques, and comprehensive performance evaluation.
## Dataset
Source: MNIST (Modified National Institute of Standards and Technology) Dataset
Training Samples: 60,000 grayscale images
Test Samples: 10,000 grayscale images
Image Dimensions: 28×28 pixels
Classes: 10 digits (0-9)
## Technologies & Frameworks
Deep Learning: TensorFlow, Keras
Data Processing: NumPy, Pandas, PyTorch
Visualization: Matplotlib, Seaborn
Image Processing: scikit-image
Model Evaluation: scikit-learn
## CNN Architecture
The model consists of:
3 Convolutional Layers with ReLU activation (32, 64, 64 filters)
2 Max Pooling Layers (2×2)
Flatten Layer (576 features)
Dense Layer (64 neurons with ReLU)
Dropout Layer (50% rate for regularization)
Output Layer (10 neurons with Softmax activation)
Total Parameters: 93,322 trainable parameters
## Project Pipeline
### 1. Data Exploration & Visualization
Dataset loading using TensorFlow/Keras
Statistical analysis of image dimensions
Visualization of sample digits with labels
### 2. Data Preprocessing
Image reshaping to (28, 28, 1) format
Pixel normalization using mean subtraction and standard deviation scaling
One-hot encoding of labels for categorical classification
Train-validation-test split (80%-10%-10%)
Conversion between NumPy arrays, PyTorch tensors, and TensorFlow tensors
### 3. Model Development
Sequential CNN model construction
Layer-by-layer architecture design
Model compilation with Adam optimizer
Categorical cross-entropy loss function
### 4. Training & Optimization
Initial training with 10 epochs and batch size of 64
Early stopping implementation to prevent overfitting (patience=3)
Monitoring validation loss for optimal checkpoint saving
### 5. Model Evaluation
Performance metrics: Accuracy, Precision, Recall, F1-Score
Confusion matrix analysis
Training/validation loss and accuracy curves
Test set evaluation
### 6. Advanced Techniques
Dropout Regularization (0.5 rate)
Data Augmentation:
Random rotation (±10°)
Width/height shift (10%)
Zoom transformation (10%)
## Results
Test Accuracy: 98.33%
Test Loss: 0.0594
Precision: 98.36%
Recall: 98.33%
F1-Score: 98.32%
## Key Features
Complete data preprocessing pipeline with multiple normalization techniques
Flexible tensor conversion (NumPy ↔ PyTorch ↔ TensorFlow)
Early stopping mechanism for efficient training
Comprehensive evaluation with multiple performance metrics
Data augmentation for improved generalization
Dropout regularization to prevent overfitting
Visual analysis of model performance with training curves
## Model Insights
The model successfully learns complex patterns in handwritten digits
Regularization techniques effectively prevent overfitting
Data augmentation improves model robustness to variations
The confusion matrix shows minimal misclassification across all digit classes
