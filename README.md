# 🛡️ TrueFace-API: Face Liveness Detection System

## Overview
A robust binary classification system designed to detect presentation attacks (spoofing) in real-time. Built using MobileNetV2 and trained on a dataset of 8,000+ images.

## 🚀 Key Features
- **Real-time Detection**: Optimized for high-speed inference.
- **Transfer Learning**: Utilizes MobileNetV2 for superior feature extraction.
- **Robustness**: Trained to distinguish between real human faces and high-resolution photos/screens.

## 🛠️ Technical Stack
- **Framework**: TensorFlow/Keras
- **Base Model**: MobileNetV2 (ImageNet weights)
- **Pre-processing**: OpenCV (Haar Cascades for face localization)
- **Input Shape**: 224x224x3

## 📊 Performance
- **Validation Accuracy**: ~92%
- **Loss**: Binary Crossentropy

## 📂 Repository Structure
- `liveness_model_v3_robust.keras`: The serialized Keras model.
- `liveness_detection.py`: Core inference logic.
- `README.md`: Project documentation.
