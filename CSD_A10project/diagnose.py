import sys
try:
    import flask
    print(f"Flask: {flask.__version__}")
    import flask_mail
    print("Flask-Mail: Installed")
    import shap
    print(f"SHAP: {shap.__version__}")
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    print("Matplotlib: Installed and Agg backend set")
    import joblib
    print("Joblib: Installed")
    import pandas as pd
    print("Pandas: Installed")
    import numpy as np
    print("Numpy: Installed")
    import ollama
    print("Ollama: Installed")
    import xgboost
    print("XGBoost: Installed")
    print("DIAGNOSTIC SUCCESS")
except Exception as e:
    print(f"DIAGNOSTIC FAILURE: {str(e)}")
    sys.exit(1)
