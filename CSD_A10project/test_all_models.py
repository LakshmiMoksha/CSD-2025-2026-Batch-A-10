import joblib
import pandas as pd
import numpy as np
import os
import sys

# Define models to test
models_paths = {
    'DecisionTree': 'decision_tree_model.pkl',
    'RandomForest': 'random_forest_model.pkl',
    'MLP': 'mlp_model.pkl',
    'XGBoost': 'xgboost_model.pkl',
    'AdaBoost': 'adaboost_model.pkl',
    'Autoencoder': 'autoencoder_model.pkl'
}

DATA_PATH = 'WSN-DS.csv'

def test_models():
    print("="*60)
    print("VERIFYING ALL MODELS")
    print("="*60)

    # 1. Load Encoders
    print("\n[ Step 1 ] Loading Encoders...")
    encoders = {}
    if os.path.exists('encoders.pkl'):
        try:
            encoders = joblib.load('encoders.pkl')
            print(f"  ✓ Loaded encoders for: {list(encoders.keys())}")
        except Exception as e:
            print(f"  X Failed to load encoders: {e}")
    else:
        print("  ! No encoders.pkl found (Will assume numeric input)")

    # 2. Load Sample Data
    print("\n[ Step 2 ] Loading Sample Data...")
    try:
        df = pd.read_csv(DATA_PATH, nrows=10)
        print("  ✓ Loaded sample rows.")
    except Exception as e:
        print(f"  X Failed to load data: {e}")
        return

    # Prepare input vector (feature columns only)
    # We need to handle 'Attack type' column drop
    if 'Attack type' in df.columns:
        X_raw = df.drop('Attack type', axis=1)
        # Verify columns match expected 18 features (ignoring names for a moment, just count)
        # Expected from previous debugging: 18 features
    else:
        X_raw = df.iloc[:, :18]

    # 3. Test Each Model
    print("\n[ Step 3 ] Testing Models...")
    
    # Preprocess Logic (Mimicking app.py)
    # We need to replicate the logic: use encoder if label exists, else to_numeric
    
    X_processed = X_raw.copy()
    clean_enc_keys = {k.strip(): k for k in encoders.keys()}
    
    for col in X_processed.columns:
        clean_col = col.strip()
        if clean_col in clean_enc_keys:
             le = encoders[clean_enc_keys[clean_col]]
             # transform safety
             X_processed[col] = X_processed[col].apply(lambda x: transform_label(le, x))
        else:
             X_processed[col] = pd.to_numeric(X_processed[col], errors='coerce').fillna(0)

    # Convert to standard array for prediction
    # Note: Some models might want dataframe with names, others just array.
    # XGBoost usually likes names if trained with names, or array if not.
    # The app passes `[abc]` which is a list (array-like).
    
    sample_input = X_processed.iloc[0:5] # Take 5 samples
    
    success_count = 0
    total_models = len(models_paths)

    for name, path in models_paths.items():
        print(f"\n  Testing {name}...")
        if not os.path.exists(path):
            print(f"    X Model file not found: {path}")
            continue
            
        try:
            model = joblib.load(path)
            # Predict
            try:
                # App logic: model.predict([abc]) -> single sample
                # We test batch here for efficiency
                preds = model.predict(sample_input)
                print(f"    ✓ Loaded & Predicted: {preds}")
                success_count += 1
            except Exception as e:
                 # Try passing as values only (some sklearn versions/models might be picky)
                 try:
                     preds = model.predict(sample_input.values)
                     print(f"    ✓ Loaded & Predicted (values only): {preds}")
                     success_count += 1
                 except Exception as e2:
                     print(f"    X Prediction failed: {e}")
                     print(f"    X Prediction (values) failed: {e2}")

        except Exception as e:
            print(f"    X Failed to load model: {e}")

    print("\n" + "="*60)
    print(f"SUMMARY: {success_count}/{total_models} models working correctly.")
    print("="*60)

def transform_label(le, val):
    try:
        return le.transform([str(val)])[0]
    except:
        return 0

if __name__ == "__main__":
    test_models()
