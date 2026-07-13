"""
Launch Failure — LSTM Model
Deep learning sequence classifier for launch failure prediction.
"""

import os
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def build_lstm(input_shape):
    """Build LSTM model: LSTM(128) → Dropout → LSTM(64) → Dense(1 sigmoid)."""
    import tensorflow as tf
    tf.get_logger().setLevel("ERROR")
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout

    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape),
        Dropout(0.3),
        LSTM(64),
        Dropout(0.3),
        Dense(32, activation="relu"),
        Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def train_lstm(X_train, y_train, X_test, y_test, model_dir):
    """Train LSTM model on launch data (reshaped to sequences)."""
    print("\n[LaunchFailure] Training LSTM Model...")
    import tensorflow as tf
    tf.get_logger().setLevel("ERROR")

    # Reshape to 3D for LSTM: (samples, timesteps=1, features)
    if len(X_train.shape) == 2:
        X_train_3d = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
        X_test_3d = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
    else:
        X_train_3d = X_train
        X_test_3d = X_test

    model = build_lstm((X_train_3d.shape[1], X_train_3d.shape[2]))

    # Train with early stopping
    from tensorflow.keras.callbacks import EarlyStopping
    es = EarlyStopping(monitor="loss", patience=10, restore_best_weights=True)

    model.fit(
        X_train_3d, y_train,
        epochs=50,
        batch_size=max(1, len(X_train) // 4),
        validation_split=0.0,  # Too little data for val split
        callbacks=[es],
        verbose=0
    )

    # Predictions
    y_prob = model.predict(X_test_3d, verbose=0).flatten()
    y_pred = (y_prob > 0.5).astype(int)

    from sklearn.metrics import accuracy_score
    acc = accuracy_score(y_test, y_pred)
    print(f"  LSTM Test Accuracy: {acc:.4f}")

    # Save model
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "lstm_launch.keras")
    model.save(model_path)
    print(f"  Model saved to: {model_path}")

    return model, y_prob


def load_model(model_dir):
    """Load trained LSTM model."""
    import tensorflow as tf
    tf.get_logger().setLevel("ERROR")
    return tf.keras.models.load_model(os.path.join(model_dir, "lstm_launch.keras"))
