# 🛡️ TrueFace-API: Professional Face Liveness Detection

## 📝 Project Scope
An end-to-end computer vision pipeline to distinguish real human faces from print/screen spoofs.

## 📊 Model Performance Evolution
- **Phase 1 (Baseline)**: Initial model trained on 25 images. 
  - Accuracy: 50% (Biased/Small Dataset)
  - Status: Biased (Predicted 'REAL' for all inputs).
- **Phase 2 (Robust)**: Retrained using MobileNetV2 on 8,000+ Kaggle images.
  - **Validation Accuracy: ~92% (Robust MobileNetV2)**
  - Status: Production Ready.

## ⚙️ Engineering Specs
- **Backbone**: MobileNetV2 (Transfer Learning)
- **Input**: 224x224 RGB Images
- **Inference**: OpenCV-based face cropping + Keras Model

## 📂 Contents
- `main.py`: Full model architecture and inference logic.
- `liveness_model_v3_robust.keras`: Pre-trained robust model weights.
- `requirements.txt`: Environment dependencies.
