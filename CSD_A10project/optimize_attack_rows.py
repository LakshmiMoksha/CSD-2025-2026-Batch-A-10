import pandas as pd
import joblib
import numpy as np
import os
import warnings

warnings.filterwarnings("ignore")

# Configuration
GENERATED_DIR = 'generated_attack_files'
MODEL_PATHS = {
    'RandomForest': 'random_forest_model.pkl',
    'XGBoost': 'xgboost_model.pkl',
    'DecisionTree': 'decision_tree_model.pkl',
    'MLP': 'mlp_model.pkl'
}

# Targets to optimize (Filename -> Target Label ID)
TARGETS = {
    'Grayhole.csv': 1,
    'Blackhole.csv': 2,
    'TDMA.csv': 3,
    'Flooding.csv': 4,
    'Normal.csv': 0
}

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

def get_fitness(row_values, models, target_label):
    # Fitness = Average Probability of Target Label
    probs = []
    for model in models.values():
        try:
            if hasattr(model, "predict_proba"):
                p = model.predict_proba(row_values.reshape(1, -1))[0]
                if len(p) > target_label:
                    probs.append(p[target_label])
                else:
                    # Some models might not have all classes if they weren't trained on them
                    probs.append(0)
            else:
                # If no proba, rely on hard predict
                pred = model.predict(row_values.reshape(1, -1))[0]
                probs.append(1.0 if pred == target_label else 0.0)
        except:
            probs.append(0) 
    
    if not probs:
        return 0
    return sum(probs) / len(probs)

def optimize_file(filename, target_label, models):
    filepath = os.path.join(GENERATED_DIR, filename)
    if not os.path.exists(filepath):
        print(f"Skipping {filename} (not found)")
        return

    print(f"\nOptimizing {filename} for Label {target_label}...")
    df = pd.read_csv(filepath)
    
    # Start with the best existing row (if any) or just the first one
    # We can try to find a seed row that is "closest" first
    best_init_idx = 0
    best_init_fitness = -1
    
    # Check first 100 rows for a good seed
    check_limit = min(len(df), 100)
    print(f"  Scanning first {check_limit} rows for best seed...")
    for i in range(check_limit):
        feat = df.iloc[i, :18].apply(pd.to_numeric, errors='coerce').fillna(0).values
        f = get_fitness(feat, models, target_label)
        if f > best_init_fitness:
            best_init_fitness = f
            best_init_idx = i
            
    print(f"  Best seed at index {best_init_idx} with fitness {best_init_fitness:.4f}")
    
    start_features = df.iloc[best_init_idx, :18].apply(pd.to_numeric, errors='coerce').fillna(0).values
    current_features = start_features.copy()
    current_fitness = best_init_fitness
    
    # Hill Climbing
    iterations = 3000
    step_size = 0.5 # Smaller step size for fine tuning
    
    for i in range(iterations):
        noise = np.random.normal(0, step_size, size=current_features.shape)
        candidate = current_features + noise
        candidate = np.maximum(candidate, 0) # Assume non-negative features
        
        fitness = get_fitness(candidate, models, target_label)
        
        if fitness > current_fitness:
            current_fitness = fitness
            current_features = candidate
            # print(f"    Iter {i}: Fitness -> {current_fitness:.4f}")
            
            if current_fitness > 0.95:
                print(f"  Reached high confidence ({current_fitness:.4f}) at iter {i}")
                break
                
    print(f"  Final Fitness: {current_fitness:.4f}")
    
    # Verify predictions
    votes = 0
    final_preds = {}
    for name, model in models.items():
        p = model.predict(current_features.reshape(1, -1))[0]
        final_preds[name] = p
        if p == target_label:
            votes += 1
            
    print(f"  Final Consensus: {final_preds}")
    
    if votes > 0:
        print(f"  Updating {filename} with optimized row...")
        df.iloc[0, :18] = current_features
        df.to_csv(filepath, index=False)
        print("  Done.")
    else:
        print("  Optimization failed to find a valid row.")

def main():
    models = load_models()
    if not models:
        print("No models loaded.")
        return

    for filename, target_id in TARGETS.items():
        optimize_file(filename, target_id, models)

if __name__ == "__main__":
    main()
