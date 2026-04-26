import cv2
import numpy as np
import tensorflow as tf

def load_liveness_model(model_path='liveness_model_v3_robust.keras'):
    return tf.keras.models.load_model(model_path)

def predict_liveness(face_img, model):
    face_img = cv2.resize(face_img, (224, 224)) / 255.0
    face_img = np.expand_dims(face_img, axis=0)
    prediction = model.predict(face_img, verbose=0)[0][0]
    label = 'SPOOF' if prediction > 0.5 else 'REAL'
    confidence = prediction if prediction > 0.5 else 1 - prediction
    return label, confidence
