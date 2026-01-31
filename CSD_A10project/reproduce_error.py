import joblib
import pandas as pd
import numpy as np
import traceback

try:
    print("Loading XGBoost model...")
    model = joblib.load('xgboost_model.pkl')
    print("Model loaded.")

    # Case 1: Numeric input (should work)
    print("\nCase 1: Numeric input")
    numeric_input = [0, 0, 1, 1, 0.5, 1, 1, 1, 1, 1, 1, 1, 0.1, 0.1, 0.1, 0.1, 1, 0.01]
    try:
        model.predict([numeric_input])
        print("Success for numeric input")
    except Exception as e:
        print(f"Failed for numeric input: {e}")

    # Case 2: String input (should fail with Unicode-2)
    print("\nCase 2: String input")
    string_input = [0, 0, 1, "Normal", 0.5, 1, 1, 1, 1, 1, 1, 1, 0.1, 0.1, 0.1, 0.1, 1, 0.01]
    try:
        model.predict([string_input])
        print("Success for string input (Unexpected)")
    except Exception as e:
        print(f"Failed for string input as expected: {e}")
        traceback.print_exc()

    # Case 3: Empty input (should fail with IndexError)
    print("\nCase 3: Empty input")
    try:
        empty_df = pd.DataFrame()
        # Simulate app.py logic
        if empty_df.empty:
             print("Detected empty dataframe (Simulated safe check)")
        else:
             print(empty_df.iloc[0])
        
        # Actual crash simulation
        print("Attempting crash...")
        print(empty_df.iloc[0])
    except IndexError as e:
         print(f"Caught expected IndexError: {e}")
    except Exception as e:
         print(f"Caught unexpected error: {e}")

except Exception as e:
    print(f"General error: {e}")
