# 🛡️ TrueFace-API: Professional Face Liveness Detection

## 📝 Project Scope
An end-to-end computer vision pipeline to distinguish real human faces from print/screen spoofs.

## ⚙️ Engineering Specs
- **Backbone**: MobileNetV2 (Transfer Learning)
- **Data**: 8,000+ Image dataset with real-time augmentation
- **Input**: 224x224 RGB Images
- **Inference**: OpenCV-based face cropping + Keras Model

## 📂 Contents
- `main.py`: Full model architecture and inference logic.
- `liveness_model_v3_robust.keras`: Pre-trained robust model weights.
