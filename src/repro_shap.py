
import os
import numpy as np
import pandas as pd
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras import layers
import shap
import matplotlib.pyplot as plt

# 1. Test XGBoost Fixes
print("\n--- Testing XGBoost SHAP Fixes ---")
X = np.random.rand(100, 5).astype(np.float32)
y = (X[:, 0] + X[:, 1] > 1).astype(int)

# Try with explicit base_score and newer SHAP API
print("Testing XGBoost with shap.Explainer...")
model = xgb.XGBClassifier(n_estimators=10, base_score=0.5)
model.fit(X, y)

try:
    explainer = shap.Explainer(model)
    shap_values = explainer(X)
    print("XGBoost shap.Explainer successful")
except Exception as e:
    print(f"XGBoost shap.Explainer failed: {e}")

try:
    print("Testing XGBoost with TreeExplainer and feature_perturbation='interventional'...")
    explainer = shap.TreeExplainer(model, feature_perturbation='interventional', data=X)
    shap_values = explainer.shap_values(X)
    print("XGBoost TreeExplainer (interventional) successful")
except Exception as e:
    print(f"XGBoost TreeExplainer (interventional) failed: {e}")

# 2. Test LSTM Fixes
print("\n--- Testing LSTM SHAP Fixes ---")
X_lstm = np.random.rand(100, 10, 5).astype(np.float32)
y_lstm = (np.mean(X_lstm, axis=(1, 2)) > 0.5).astype(int)

model_lstm = tf.keras.Sequential([
    layers.Input(shape=(10, 5)),
    layers.LSTM(16),
    layers.Dense(1, activation='sigmoid')
])
model_lstm.compile(optimizer='adam', loss='binary_crossentropy')
model_lstm.fit(X_lstm, y_lstm, epochs=1, verbose=0)

print("Testing LSTM with shap.Explainer (Permutation/Kernel)...")
try:
    # Use a small background and small test set because it's slow
    background = X_lstm[:10]
    explainer = shap.Explainer(model_lstm.predict, background)
    shap_values = explainer(X_lstm[:2])
    print("LSTM shap.Explainer successful")
except Exception as e:
    print(f"LSTM shap.Explainer failed: {e}")

print("Testing LSTM with GradientExplainer (TF 2.x compatible)...")
try:
    # GradientExplainer often works better with Keras in TF 2.x
    explainer = shap.GradientExplainer(model_lstm, X_lstm[:50])
    shap_values = explainer.shap_values(X_lstm[:2])
    print("LSTM GradientExplainer successful")
except Exception as e:
    print(f"LSTM GradientExplainer failed: {e}")
