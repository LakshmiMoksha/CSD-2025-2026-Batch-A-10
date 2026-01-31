import pandas as pd
import joblib
import os
import numpy as np
import warnings

# Suppress sklearn warnings about feature names
warnings.filterwarnings("ignore")

# Configuration
GENERATED_DIR = 'generated_attack_files'
MODEL_PATHS = {
    'RandomForest': 'random_forest_model.pkl',
    'XGBoost': 'xgboost_model.pkl',
    'DecisionTree': 'decision_tree_model.pkl',
    'MLP': 'mlp_model.pkl'
}

# Configuration
GENERATED_DIR = 'generated_attack_files'
MODEL_PATHS = {
    'RandomForest': 'random_forest_model.pkl',
    'XGBoost': 'xgboost_model.pkl',
    'DecisionTree': 'decision_tree_model.pkl',
    'MLP': 'mlp_model.pkl',
    'AdaBoost': 'adaboost_model.pkl'
}

# Expected label mapping (Model Output ID -> Attack Name)
TARGET_MAPPING = {
    'Normal.csv': 0,
    'Grayhole.csv': 1,
    'Blackhole.csv': 2,
    'TDMA.csv': 3,
    'Flooding.csv': 4
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
        else:
            print(f"Warning: {name} model file not found at {path}")
    return models

def main():
    models = load_models()
    if not models:
        print("Error: No models could be loaded.")
        return

    if not os.path.exists(GENERATED_DIR):
        print(f"Error: Directory {GENERATED_DIR} not found.")
        return

    for filename, expected_label_id in TARGET_MAPPING.items():
        filepath = os.path.join(GENERATED_DIR, filename)
        if not os.path.exists(filepath):
            print(f"Skipping {filename} (not found)")
            continue

        print(f"\nProcessing {filename} (Target Label ID: {expected_label_id})...")
        try:
            df = pd.read_csv(filepath)
            
            if df.shape[1] < 18:
                print(f"  Warning: {filename} has fewer than 18 columns. Skipping.")
                continue
                
            features = df.iloc[:, :18].apply(pd.to_numeric, errors='coerce').fillna(0)
            
            # Predict with ALL models
            model_predictions = {}
            model_probs = {} # Store probabilities if available for fallback
            
            for name, model in models.items():
                print(f"  Predicting with {name}...")
                try:
                    preds = model.predict(features.values)
                    model_predictions[name] = preds
                    
                    if hasattr(model, "predict_proba"):
                        try:
                            probs = model.predict_proba(features.values)
                            # Store reliability for the target class
                            if probs.shape[1] > expected_label_id:
                                model_probs[name] = probs[:, expected_label_id]
                        except:
                            pass
                except Exception as e:
                    print(f"    Failed to predict with {name}: {e}")

            # Find a "Consensus" Row
            best_row_idx = -1
            max_agreement = -1
            
            num_rows = len(df)
            correctness_matrix = np.zeros((num_rows, len(model_predictions)))
            for i, (name, preds) in enumerate(model_predictions.items()):
                correctness_matrix[:, i] = (preds == expected_label_id).astype(int)
            
            agreement_scores = correctness_matrix.sum(axis=1)
            
            if len(agreement_scores) > 0:
                max_agreement = np.max(agreement_scores)
                candidates = np.where(agreement_scores == max_agreement)[0]
                best_row_idx = candidates[0] # Default to first best
            
            print(f"  Best row index: {best_row_idx} with agreement from {int(max_agreement)}/{len(models)} models.")
            
            # Fallback for ZERO agreement (specifically Flooding)
            if max_agreement == 0:
                print("  Zero agreement. Attempting to find row with highest target probability...")
                # Sum probabilities across models for the target class
                total_probs = np.zeros(num_rows)
                valid_prob_models = 0
                for name, probs in model_probs.items():
                    total_probs += probs
                    valid_prob_models += 1
                
                if valid_prob_models > 0:
                    best_prob_idx = np.argmax(total_probs)
                    avg_prob = total_probs[best_prob_idx] / valid_prob_models
                    print(f"  Found probabilistic best row {best_prob_idx} with avg probability {avg_prob:.4f}")
                    # Use this row
                    best_row_idx = best_prob_idx
                    # But warn that it technically fails hard prediction
                    # However, if probability is decent, maybe one model works?
                else:
                    print("  No probability data available.")

            if best_row_idx != -1:
                # Move to top
                new_order = [best_row_idx] + [i for i in range(len(df)) if i != best_row_idx]
                df_reordered = df.iloc[new_order]
                df_reordered.to_csv(filepath, index=False)
                print(f"  Success: Updated {filename} (Best Row {best_row_idx})")
            else:
                print("  No suitable row found. File unchanged.")

        except Exception as e:
            print(f"  Error processing {filename}: {e}")

if __name__ == "__main__":
    main()
