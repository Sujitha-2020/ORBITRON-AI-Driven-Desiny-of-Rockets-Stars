"""
Satellite Health — Autoencoder for Anomaly Detection
Architecture: Input → 64 → 32 → 16 → 32 → 64 → Output
Trained on normal operation data; reconstruction error indicates anomalies.
"""

import os
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def build_autoencoder(input_dim):
    """Build symmetric autoencoder: Input → 64 → 32 → 16 → 32 → 64 → Output."""
    import tensorflow as tf
    tf.get_logger().setLevel("ERROR")
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, Dense

    inp = Input(shape=(input_dim,))
    # Encoder
    x = Dense(64, activation="relu")(inp)
    x = Dense(32, activation="relu")(x)
    encoded = Dense(16, activation="relu")(x)
    # Decoder
    x = Dense(32, activation="relu")(encoded)
    x = Dense(64, activation="relu")(x)
    decoded = Dense(input_dim, activation="linear")(x)

    autoencoder = Model(inp, decoded, name="SatelliteAutoencoder")
    autoencoder.compile(optimizer="adam", loss="mse")
    return autoencoder


def train_autoencoder(X_train_normal, model_dir, epochs=100, batch_size=64):
    """Train autoencoder on normal operation data."""
    print("\n[SatelliteHealth] Training Autoencoder...")
    import tensorflow as tf
    tf.get_logger().setLevel("ERROR")
    from tensorflow.keras.callbacks import EarlyStopping

    model = build_autoencoder(X_train_normal.shape[1])

    es = EarlyStopping(monitor="loss", patience=10, restore_best_weights=True)
    model.fit(
        X_train_normal, X_train_normal,
        epochs=epochs,
        batch_size=min(batch_size, len(X_train_normal)),
        validation_split=0.1,
        callbacks=[es],
        verbose=0
    )

    # Compute reconstruction error on training data
    reconstructed = model.predict(X_train_normal, verbose=0)
    recon_errors = np.mean((X_train_normal - reconstructed) ** 2, axis=1)

    # Save model and threshold
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "autoencoder_satellite.keras")
    model.save(model_path)

    # Compute threshold: mean + 2 * std
    threshold = recon_errors.mean() + 2 * recon_errors.std()
    np.save(os.path.join(model_dir, "ae_threshold.npy"), threshold)

    print(f"  Reconstruction Error — Mean: {recon_errors.mean():.6f}, Std: {recon_errors.std():.6f}")
    print(f"  Anomaly Threshold: {threshold:.6f}")
    print(f"  Model saved to: {model_path}")

    return model, threshold, recon_errors


def load_model(model_dir):
    """Load trained autoencoder and threshold."""
    import tensorflow as tf
    tf.get_logger().setLevel("ERROR")
    model = tf.keras.models.load_model(os.path.join(model_dir, "autoencoder_satellite.keras"))
    threshold = float(np.load(os.path.join(model_dir, "ae_threshold.npy")))
    return model, threshold
