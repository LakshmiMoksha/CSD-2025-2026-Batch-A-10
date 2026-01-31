import pandas as pd
import numpy as np
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
from sklearn.ensemble import RandomForestClassifier

FEATURE_NAMES = [
    'ID', 'Time', 'Is_CH', 'Who_CH', 'Dist_to_CH', 'ADV_S', 'ADV_R', 
    'JOIN_S', 'JOIN_R', 'SCH_S', 'SCH_R', 'Rank', 'DATA_S', 'DATA_R', 
    'Sent_to_BS', 'Dist_to_BS', 'Send_Code', 'Energy'
]

def generate_shap_plot(model, features, predicted_class_idx):
    try:
        print("Starting SHAP generation...")
        X = pd.DataFrame([features], columns=FEATURE_NAMES)
        bg = X 
        
        print(f"Model: {type(model)}")
        print(f"Predicted Class Index: {predicted_class_idx}")
        
        # Test logic
        if hasattr(model, 'predict_proba'):
            print("Using predict_proba explainer...")
            explainer = shap.Explainer(model.predict_proba, bg)
            shap_values = explainer(X)
            print(f"shap_values type: {type(shap_values)}")
            # Handle Explanation object
            if isinstance(shap_values, shap.Explanation):
                print("Processing Explanation object...")
                if len(shap_values.shape) == 3: # (samples, features, classes)
                    single_shap_values = shap_values[0, :, predicted_class_idx]
                else:
                    single_shap_values = shap_values[0]
            else:
                print(f"Unknown shap_values type: {type(shap_values)}")
                single_shap_values = shap_values[0]
        else:
            print("Using predict explainer...")
            explainer = shap.Explainer(model.predict, bg)
            shap_values = explainer(X)
            single_shap_values = shap_values[0]
        
        print(f"single_shap_values type: {type(single_shap_values)}")
        
        plt.figure(figsize=(10, 4))
        plt.style.use('dark_background')
        
        print("Rendering bar plot...")
        # Fallback to manual bar plot if shap.plots.bar fails
        try:
            shap.plots.bar(single_shap_values, show=False)
        except Exception as e:
            print(f"shap.plots.bar failed: {e}. Falling back to manual plt.bar")
            # If it's a numpy array, plot it manually
            if hasattr(single_shap_values, 'values'):
                vals = single_shap_values.values
                names = single_shap_values.feature_names
            else:
                vals = single_shap_values
                names = FEATURE_NAMES
            plt.barh(names, vals)
            
        plt.title(f"Model Reason: Why Class {predicted_class_idx}?", color='#a855f7', pad=20)
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', transparent=True)
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        print("SHAP SUCCESS")
        return img_base64
    except Exception as e:
        print(f"SHAP FAILURE: {e}")
        import traceback
        traceback.print_exc()
        return None

# Dummy data and model
X_train = np.random.rand(100, 18)
y_train = np.random.randint(0, 5, 100)
model = RandomForestClassifier().fit(X_train, y_train)
sample_features = X_train[0].tolist()

res = generate_shap_plot(model, sample_features, int(y_train[0]))
if res:
    print(f"Image base64 length: {len(res)}")
else:
    print("Failed to generate image.")
