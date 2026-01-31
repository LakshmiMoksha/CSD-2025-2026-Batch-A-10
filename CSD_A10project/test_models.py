import joblib
import numpy as np

print("Testing model compatibility...")

models_to_test = {
    'LogisticRegression': 'logistic_model.pkl',
    'AdaBoost': 'adaboost_model.pkl',
    'Stacking': 'stacking_model.pkl'
}

test_input = np.random.rand(1, 18)

for name, path in models_to_test.items():
    try:
        print(f"\n Testing {name}...")
        model = joblib.load(path)
        result = model.predict(test_input)
        print(f"✓ {name} works! Result: {result}")
    except Exception as e:
        print(f"✗ {name} FAILED: {e}")
