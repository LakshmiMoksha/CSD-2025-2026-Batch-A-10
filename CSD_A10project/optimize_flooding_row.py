import pandas as pd
import joblib
import numpy as np
import os
import random

# Configuration
GENERATED_DIR = 'generated_attack_files'
FLOODING_FILE = 'Flooding.csv'
MODEL_PATHS = {
    'RandomForest': 'random_forest_model.pkl',
    'XGBoost': 'xgboost_model.pkl' # Focus on these two as they are most likely used
}
TARGET_LABEL = 4 # Flooding

def load_models():
    models = {}
    for name, path in MODEL_PATHS.items():
        if os.path.exists(path):
            try:
                print(f"Loading {name} from {path}...")
                models[name] = joblib.load(path)
            except Exception as e:
                print(f"Failed to load {name}: {e}")
    return models

def get_fitness(row_values, models):
    # Fitness = Average Probability of Target Label
    probs = []
    for model in models.values():
        try:
            p = model.predict_proba(row_values.reshape(1, -1))[0]
            if len(p) > TARGET_LABEL:
                probs.append(p[TARGET_LABEL])
            else:
                probs.append(0)
        except:
            probs.append(0) # If predict_proba fails
    
    if not probs:
        return 0
    return sum(probs) / len(probs)

def main():
    models = load_models()
    if not models:
        print("No models loaded.")
        return

    filepath = os.path.join(GENERATED_DIR, FLOODING_FILE)
    if not os.path.exists(filepath):
        print(f"{filepath} not found.")
        return

    df = pd.read_csv(filepath)
    if df.shape[1] < 18:
        print("Not enough columns.")
        return

    # Start with the first row (which we already moved to top as 'best guess')
    # Or maybe try a few random rows as seeds
    best_row_idx = 0 
    
    # Extract features (18 numeric)
    # The dataframe might have 'ID' etc which are removed before prediction in app.py
    # But wait, app.py removes nothing?
    # app.py: features_data = csv_data.iloc[:, :18] ...
    # My generated CSVs have headers: ID, Time, Is_CH, ...
    # So the values in df are correct.
    
    start_features = df.iloc[best_row_idx, :18].apply(pd.to_numeric, errors='coerce').fillna(0).values
    
    current_features = start_features.copy()
    current_fitness = get_fitness(current_features, models)
    
    print(f"Initial Fitness (Avg Prob of Flooding): {current_fitness:.4f}")
    
    # Hill Climbing / Annealing
    iterations = 2000
    step_size = 1.0 # Standard deviation for noise
    
    print("Optimizing...")
    for i in range(iterations):
        # Create candidate
        noise = np.random.normal(0, step_size, size=current_features.shape)
        # Apply mask to keep some columns like ID/Time potentially fixed? 
        # Actually, ID/Time are features 0 and 1. They shouldn't matter much for attack type relative to traffic stats?
        # But let's just perturb everything slightly.
        
        candidate = current_features + noise
        # Ensure non-negative if needed? Most features look positive.
        candidate = np.maximum(candidate, 0) 
        
        fitness = get_fitness(candidate, models)
        
        if fitness > current_fitness:
            current_fitness = fitness
            current_features = candidate
            print(f"  Iter {i}: Improved Fitness -> {current_fitness:.4f}")
            
            # Check hard prediction
            votes = 0
            for name, model in models.items():
                if model.predict(current_features.reshape(1, -1))[0] == TARGET_LABEL:
                    votes += 1
            
            if votes == len(models):
                print(f"  SUCCESS! Found robust input at iter {i}")
                break
            
            if current_fitness > 0.6 and votes > 0:
                 print(f"  Partial Success (Prob {current_fitness:.2f}, Votes {votes})")
                 # We can break early if good enough
                 if current_fitness > 0.8:
                     break

    print(f"Final Fitness: {current_fitness:.4f}")
    
    # Update DataFrame
    # verification
    votes = 0
    final_preds = {}
    for name, model in models.items():
        p = model.predict(current_features.reshape(1, -1))[0]
        final_preds[name] = p
        if p == TARGET_LABEL:
            votes += 1
            
    print(f"Final Predictions: {final_preds}")
    
    if votes > 0:
        print("Updating CSV with optimized row...")
        # Update row 0
        df.iloc[0, :18] = current_features
        df.to_csv(filepath, index=False)
        print("Done.")
    else:
        print("Failed to find a valid Flooding row. Optimization failed.")

if __name__ == "__main__":
    main()
