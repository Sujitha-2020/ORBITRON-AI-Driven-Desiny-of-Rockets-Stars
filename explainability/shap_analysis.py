"""
ORBITRON — SHAP Explainability Module
TreeExplainer for XGBoost, DeepExplainer for LSTM.
Generates feature importance and summary plots.
"""

import os
import numpy as np
import warnings
warnings.filterwarnings("ignore")

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def explain_xgboost(model, X_data, feature_names, output_dir):
    """Generate SHAP explanations for XGBoost model."""
    print("\n[Explainability] Generating SHAP analysis for XGBoost...")
    import shap
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)

    try:
        # 1. Try standard TreeExplainer
        import pandas as pd
        if not isinstance(X_data, pd.DataFrame):
            X_df = pd.DataFrame(X_data, columns=feature_names)
        else:
            X_df = X_data

        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_df)
        except Exception as e:
            if "could not convert string to float" in str(e).lower():
                print(f"  Standard TreeExplainer failed due to XGBoost base_score bug. Trying KernelExplainer fallback...")
                # Fallback to KernelExplainer for XGBoost 2.0+ base_score issue
                # Use a small sample for speed
                bg_data = shap.sample(X_df, min(50, len(X_df)))
                explainer = shap.KernelExplainer(model.predict_proba, bg_data)
                shap_values = explainer.shap_values(X_df[:20]) # Limit to 20 samples for speed in fallback
                X_df = X_df[:20] # Sync X_df with sampled shap_values
            else:
                raise e

        # Handle multi-output (classification)
        if isinstance(shap_values, list):
            # For binary classification, shap_values might be a list [class0, class1]
            shap_output = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        else:
            shap_output = shap_values

        # Summary plot
        plt.figure(figsize=(12, 6))
        shap.summary_plot(shap_output, X_df, feature_names=feature_names, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "xgboost_shap_summary.png"), dpi=150, bbox_inches="tight")
        plt.close()

        # Dependence plots for top features
        # Ensure shap_output is a numpy array for indexing
        shap_output_np = np.array(shap_output)
        mean_abs_shap = np.abs(shap_output_np).mean(0)
        top_indices = np.argsort(mean_abs_shap)[::-1][:min(3, len(mean_abs_shap))]
        
        for idx in top_indices:
            try:
                feature_name = feature_names[idx]
                plt.figure(figsize=(10, 6))
                # For dependence plot, X must match shap_values rows
                shap.dependence_plot(int(idx), shap_output_np, X_df, feature_names=feature_names, show=False)
                plt.tight_layout()
                safe_name = feature_name.replace(" ", "_").lower()
                plt.savefig(os.path.join(output_dir, f"xgboost_shap_dependence_{safe_name}.png"), 
                            dpi=120, bbox_inches="tight")
                plt.close()
            except Exception as dep_e:
                print(f"    [Warning] Dependence plot for {feature_names[idx]} failed: {dep_e}")
                plt.close()

        print(f"  SHAP plots saved to: {output_dir}")
        return shap_output_np

    except Exception as e:
        print(f"  SHAP analysis failed: {e}")
        # Create simple feature importance plot as fallback
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            plt.figure(figsize=(12, 6))
            plt.title("Feature Importance (Gini)")
            plt.barh(range(len(indices)), importances[indices])
            plt.yticks(range(len(indices)),
                       [feature_names[i] if i < len(feature_names) else f"f{i}" for i in indices])
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "xgboost_feature_importance.png"), dpi=150, bbox_inches="tight")
            plt.close()
            print(f"  Fallback feature importance saved.")
        return None



def explain_lstm(model, X_data, feature_names, output_dir):
    """Generate SHAP explanations for LSTM model using DeepExplainer."""
    print("\n[Explainability] Generating SHAP analysis for LSTM...")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)

    try:
        import shap
        import tensorflow as tf
        tf.get_logger().setLevel("ERROR")

        # Reshape for LSTM
        if len(X_data.shape) == 2:
            X_3d = X_data.reshape(X_data.shape[0], 1, X_data.shape[1])
        else:
            X_3d = X_data

        # Use a subset as background
        bg_size = min(100, len(X_3d))
        background = X_3d[:bg_size]

        print("  Using GradientExplainer for TensorFlow model compatibility...")
        explainer = shap.GradientExplainer(model, background)
        
        # Explain a subset for performance
        explain_size = min(50, len(X_3d))
        shap_values = explainer.shap_values(X_3d[:explain_size])

        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        # Reshape for plotting (Flatten temporal dimension for summary)
        N_samples = shap_values.shape[0]
        shap_2d = shap_values.reshape(N_samples, -1)
        X_2d = X_3d[:N_samples].reshape(N_samples, -1)
        
        if X_3d.shape[1] > 1:
            extended_feature_names = [f"{fn}_t{t}" for t in range(X_3d.shape[1]) for fn in feature_names]
        else:
            extended_feature_names = feature_names

        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_2d, X_2d, feature_names=extended_feature_names, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "lstm_shap_summary.png"), dpi=150, bbox_inches="tight")
        plt.close()

        print(f"  LSTM SHAP plots saved to: {output_dir}")
        return shap_values

    except Exception as e:
        print(f"  LSTM SHAP analysis failed: {e}")
        return None



def run_shap_analysis(xgb_model, lstm_model, X_data, feature_names, output_dir):
    """Run complete SHAP analysis for both models."""
    shap_dir = os.path.join(output_dir, "shap")
    os.makedirs(shap_dir, exist_ok=True)

    xgb_shap = explain_xgboost(xgb_model, X_data, feature_names, shap_dir)
    lstm_shap = explain_lstm(lstm_model, X_data, feature_names, shap_dir)

    return xgb_shap, lstm_shap
