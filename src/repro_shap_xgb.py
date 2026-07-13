
import os
import numpy as np
import xgboost as xgb
import shap

# Test XGBoost Booster Fix
print("\n--- Testing XGBoost Booster Fix ---")
X = np.random.rand(100, 5).astype(np.float32)
y = (X[:, 0] + X[:, 1] > 1).astype(int)

model = xgb.XGBClassifier(n_estimators=10)
model.fit(X, y)

print("Testing TreeExplainer(model.get_booster())...")
try:
    booster = model.get_booster()
    explainer = shap.TreeExplainer(booster)
    # When using booster, we might need to cast input to DMatrix or numpy without feature names if they were lost
    shap_values = explainer.shap_values(X)
    print("XGBoost TreeExplainer(booster) successful")
except Exception as e:
    print(f"XGBoost TreeExplainer(booster) failed: {e}")

print("Checking model base_score type...")
try:
    import json
    config = json.loads(model.get_booster().save_config())
    base_score = config['learner']['learner_model_param']['base_score']
    print(f"Base score in config: {base_score} (type: {type(base_score)})")
except Exception as e:
    print(f"Failed to check config: {e}")

# Another attempt: use KernelExplainer as last resort fallback
print("Testing KernelExplainer for XGBoost (Fallback)...")
try:
    explainer = shap.KernelExplainer(model.predict_proba, shap.sample(X, 10))
    shap_values = explainer.shap_values(X[:2])
    print("XGBoost KernelExplainer successful")
except Exception as e:
    print(f"XGBoost KernelExplainer failed: {e}")
