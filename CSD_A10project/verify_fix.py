import joblib
import pandas as pd
import numpy as np
import os

def test_mapping():
    print("Verifying Label Mapping Synchronization...")
    
    # Correct Mapping from train_models.py
    # 0: Normal, 1: Grayhole, 2: Blackhole, 3: TDMA, 4: Flooding
    
    mapping = {
        0: "Normal",
        1: "Grayhole",
        2: "Blackhole",
        3: "TDMA",
        4: "Flooding"
    }

    models_to_test = {
        'DecisionTree': 'decision_tree_model.pkl',
        'RandomForest': 'random_forest_model.pkl',
        'XGBoost': 'xgboost_model.pkl'
    }

    # Load specific attack files from generated_attack_files
    # Note: These files were generated from WSN-DS.csv rows filtered by original labels
    
    attack_files = {
        'Blackhole': 'generated_attack_files/Blackhole.csv',
        'Grayhole': 'generated_attack_files/Grayhole.csv',
        'Normal': 'generated_attack_files/Normal.csv',
        'tdma': 'generated_attack_files/TDMA.csv',
        'Flooding': 'generated_attack_files/Flooding.csv'
    }

    for model_name, model_path in models_to_test.items():
        if not os.path.exists(model_path):
            print(f"Skipping {model_name} (not found)")
            continue
            
        print(f"\nTesting Model: {model_name}")
        model = joblib.load(model_path)
        
        for attack_name, csv_path in attack_files.items():
            if not os.path.exists(csv_path):
                print(f"  Skipping {attack_name} CSV (not found)")
                continue
                
            df = pd.read_csv(csv_path, nrows=5)
            # Take first row features (18 columns)
            features = df.iloc[0, :18].apply(pd.to_numeric, errors='coerce').fillna(0).values
            
            # Predict
            try:
                pred = model.predict([features])[0]
            except:
                pred = model.predict(np.array([features]))[0]
                
            predicted_name = mapping.get(pred, "Unknown")
            
            status = "PASS" if predicted_name.lower() == attack_name.lower() else "FAIL"
            print(f"  - {attack_name} CSV -> Predicted ID: {pred} ({predicted_name}) [{status}]")

if __name__ == "__main__":
    test_mapping()
