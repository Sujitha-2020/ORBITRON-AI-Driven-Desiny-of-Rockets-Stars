"""
Launch Failure — XGBoost Classifier
5-fold stratified cross-validation with tuned hyperparameters.
"""

import os
import numpy as np
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score


def train_xgboost(X_train, y_train, X_test, y_test, model_dir):
    """Train XGBoost classifier with 5-fold stratified CV."""
    print("\n[LaunchFailure] Training XGBoost Classifier...")

    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        eval_metric="logloss",
        use_label_encoder=False,
        random_state=42,
        verbosity=0
    )

    # 5-fold stratified cross-validation
    # With small datasets, we adjust n_splits to avoid errors
    n_samples = len(y_train)
    n_classes_min = min(np.bincount(y_train.astype(int)))
    n_folds = min(5, n_classes_min, n_samples)
    n_folds = max(n_folds, 2)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    try:
        cv_scores = cross_val_score(model, X_train, y_train, cv=skf, scoring="accuracy")
        print(f"  CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    except Exception as e:
        print(f"  CV skipped (small dataset): {e}")

    # Train final model
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if len(np.unique(y_test)) > 1 else model.predict_proba(X_test)[:, 0]

    acc = accuracy_score(y_test, y_pred)
    print(f"  Test Accuracy: {acc:.4f}")

    try:
        auc = roc_auc_score(y_test, y_prob)
        print(f"  Test AUC: {auc:.4f}")
    except Exception:
        auc = acc

    print(classification_report(y_test, y_pred, zero_division=0))

    # Save model
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "xgboost_launch.joblib")
    joblib.dump(model, model_path)
    print(f"  Model saved to: {model_path}")

    return model, y_prob


def load_model(model_dir):
    """Load trained XGBoost model."""
    return joblib.load(os.path.join(model_dir, "xgboost_launch.joblib"))
