import joblib
import pandas as pd
import numpy as np
import os

MODEL_PATH = 'autoencoder_model.pkl'
DATA_PATH = 'WSN-DS.csv'

def verify_model():
    print(f"Checking for {MODEL_PATH}...")
    if not os.path.exists(MODEL_PATH):
        print(f"Error: {MODEL_PATH} not found.")
        return

    print(f"Loading {MODEL_PATH}...")
    try:
        model = joblib.load(MODEL_PATH)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return


    print("Loading encoders.pkl...")
    try:
        encoders = joblib.load('encoders.pkl')
        print(f"Encoders loaded: {list(encoders.keys())}")
    except:
        print("No encoders found (or file missing). Assuming numeric input.")
        encoders = {}

    print(f"Loading sample data from {DATA_PATH}...")
    try:
        df = pd.read_csv(DATA_PATH)
        if 'Attack type' in df.columns:
            X_sample = df.drop('Attack type', axis=1).iloc[0:5]
            y_sample = df['Attack type'].iloc[0:5]
        else:
            X_sample = df.iloc[0:5, :-1]
            y_sample = df.iloc[0:5, -1]
        
        X_sample_encoded = X_sample.copy()
        
        # Apply strict encoding if encoder exists, else leave as is (numeric)
        for col in X_sample_encoded.columns:
             if col in encoders:
                 le = encoders[col]
                 # Ensure we pass string to encoder as trained
                 X_sample_encoded[col] = le.transform(X_sample_encoded[col].astype(str))
             else:
                 # If no encoder, ensure numeric
                 X_sample_encoded[col] = pd.to_numeric(X_sample_encoded[col], errors='coerce').fillna(0)
        
        print("Running prediction on 5 samples...")
        predictions = model.predict(X_sample_encoded)
        print(f"Predictions: {predictions}")
        print(f"True Labels: {y_sample.tolist()}")
        
        # Mapping check (0=Normal)
        if all(p == 0 for p in predictions) and all(l == 'Normal' for l in y_sample):
             print("Verification Successful! Correctly predicted Normal.")
        else:
             print("Verification Results Mixed (Check manually).")

    except Exception as e:
        print(f"Error during verification: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify_model()
