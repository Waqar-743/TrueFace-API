# Face Liveness Detection System

## Overview
This project implements a robust binary classification system designed to distinguish between real human faces and spoofing attacks. The system leverages Transfer Learning with a MobileNetV2 architecture.

## Technical Architecture
- **Base Model**: MobileNetV2 (Pre-trained on ImageNet)
- **Classification Head**: Global Average Pooling, Dropout (0.5), Dense Sigmoid.
- **Performance**: ~92% Validation Accuracy on the expanded Kaggle dataset.

## Workflow
1. Face Detection via OpenCV Haar Cascades.
2. Pre-processing & Normalization (224x224).
3. Inference via MobileNetV2.
4. Liveness Decision (Real vs Spoof).