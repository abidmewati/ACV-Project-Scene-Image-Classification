# Comparative Analysis of Deep Learning Models for Scene Image Classification

## Project Description

This project presents a comparative analysis of six deep learning architectures for scene image classification using the Intel Image Classification dataset.

The evaluated models are:

- VGG16
- ResNet50
- DenseNet121
- InceptionV3
- MobileNetV3
- Swin Transformer

The models are implemented using PyTorch and trained using Google Colab with GPU acceleration.

---

## Dataset

Intel Image Classification Dataset

Classes:

- Buildings
- Forest
- Glacier
- Mountain
- Sea
- Street

---

## Project Structure

Dataset/
models/
Results/
training_01.py
training_02.py
train_densenet.py
train_Inception_v3.py
train_mobilenet_v3.py
train_swin_transformer.py
train_VGG_16.py
test_all_models.ipynb

---

## Requirements

Python 3.10+

PyTorch

Torchvision

NumPy

Pandas

Matplotlib

Scikit-learn

Pillow

OpenPyXL

---

## Running the Project

### Training

Run the required training script.

Example:

python train_mobilenet_v3.py

### Testing

Open

test_all_models.ipynb

Run all notebook cells.

The notebook automatically evaluates all six trained models.

---

## Outputs

The project generates:

- Accuracy
- Precision
- Recall
- F1 Score
- Classification Report
- Confusion Matrix
- Performance Comparison Graphs
- Excel Summary

---

## Hardware

Google Colab

Tesla T4 GPU

---
## Repository Contents

This repository contains the complete source code, testing notebook, experimental results, and documentation for the project.

The Intel Image Classification dataset is publicly available and is therefore not included in this repository.

The trained model weight files (`.pth`) are not included because of their large size. The repository contains all scripts required to reproduce the experiments.


## Author

Abid Siddique

Advance Computer Vision Project
