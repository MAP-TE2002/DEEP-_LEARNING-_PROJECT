# DEEP-LEARNING-PROJECT
COMPANY : CODTECH IT SOLUTIONS
NAME : MANDRAJULA ARUN PRABHU TEJA
INTERN ID : CT06DF375
DOMAIN : DATA SCIENCE
DURATION : 6 WEEKS
MENTOR : NEELA SANTHOSH KUMAR

🧠 Problem Statement: Deep Learning for Image Classification using PyTorch

Title: Image Classification on CIFAR-10 using a Convolutional Neural Network (CNN)


Introduction:

Image classification is one of the foundational tasks in computer vision. It involves assigning a label or category to an input image from a predefined set of classes. With the growth of deep learning, convolutional neural networks (CNNs) have emerged as the most powerful and effective architectures for image classification tasks. They automatically learn spatial hierarchies of features from input images and have outperformed traditional machine learning approaches in accuracy and scalability.

In this project, we implement a CNN using PyTorch to classify images from the CIFAR-10 dataset. CIFAR-10 is a widely-used benchmark dataset consisting of 60,000 color images (32x32 pixels) categorized into 10 different classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck. The dataset is balanced with 6,000 images per class, split into 50,000 for training and 10,000 for testing.

Problem Definition:

Given an input image from the CIFAR-10 dataset, the goal is to develop a deep learning model that can accurately predict which of the 10 categories the image belongs to. The task is defined as a multi-class classification problem. The complexity of the dataset arises from the small image size, variation in orientation, lighting conditions, background noise, and similarity among certain classes (e.g., cat vs. dog, truck vs. automobile).

The model must:

Ingest image data.

Learn discriminative features through convolutional layers.

Classify the input into one of 10 classes.

Provide reliable evaluation metrics including accuracy and ROC curves.

Objectives :

This project aims to:

Load and preprocess the CIFAR-10 dataset using torchvision.

Define a CNN architecture suitable for small-sized image classification.

Train the network using CrossEntropyLoss and the Adam optimizer.

Evaluate the model’s performance on unseen test data.

Visualize ROC curves for all 10 classes to understand classification quality across classes.

Model Architecture :

The CNN implemented has the following structure:

Convolution Layer 1: 32 filters, 3x3 kernel, ReLU activation

Convolution Layer 2: 64 filters, 3x3 kernel, ReLU activation

MaxPooling: 2x2 pooling applied after each convolution block

Fully Connected Layer 1: 512 neurons, ReLU activation

Output Layer: 10 neurons (for 10 classes)

This network is simple yet powerful enough to extract useful features for classifying CIFAR-10 images with high accuracy.

Evaluation Metrics :

The model is evaluated using:

Test Accuracy: Measures the percentage of correctly classified images.

ROC Curve and AUC Score: Visual diagnostics for each class showing trade-off between true positive and false positive rates.

Challenges :

Low Image Resolution: 32x32 images are small, which limits the amount of spatial information.

Class Confusion: Similar-looking classes (like cats and dogs) can be difficult to differentiate.

Overfitting: With limited data and a powerful model, overfitting is a concern, hence requiring careful regularization or data augmentation in advanced setting

Tools & Technologies Used :

Python: Core language

PyTorch: Deep learning framework used to build and train the model

Torchvision: Utility for dataset handling and transformations

Matplotlib / Sklearn: Used for plotting ROC curves and computing AUC

Applications :

Autonomous Vehicles: Object classification in camera feeds.

Medical Imaging: Classification of image-based diagnoses.

Security Systems: Surveillance image categorization.

Retail: Automated product classification in e-commerce platforms.

Conclusion:

This project demonstrates the implementation of a deep learning model from scratch using PyTorch to solve a real-world image classification problem. The CNN achieves competitive accuracy and is a strong foundation for further experimentation, such as adding more convolutional layers, dropout, batch normalization, or implementing data augmentation for better generalization.

