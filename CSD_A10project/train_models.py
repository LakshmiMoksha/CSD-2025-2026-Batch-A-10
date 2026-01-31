import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, StackingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import os

print("Loading dataset...")
df = pd.read_csv('WSN-DS.csv')

# Preprocessing: Ensure column names are stripped of white spaces for internal logic
# But we must keep them as they are in the CSV if the app uses indices.
# The app uses df.drop('Attack type', axis=1) which works because 'Attack type' has no spaces.

# Mapping labels to match app.py expectations:
# 0 -> normal
# 1 -> Grayhole
# 2 -> Blackhole
# 3 -> tdma
# 4 -> Flooding
label_map = {
    'Normal': 0,
    'Grayhole': 1,
    'Blackhole': 2,
    'TDMA': 3,
    'Flooding': 4
}

print("Mapping labels...")
df['Attack type'] = df['Attack type'].map(label_map)

# Handle potential NaN or missing mappings
if df['Attack type'].isnull().any():
    print("Warning: Some labels were not mapped correctly. Checking unique values...")
    print(df[df['Attack type'].isnull()]['Attack type'].unique())
    # Fill with 0 as fallback or drop
    df = df.dropna(subset=['Attack type'])

X = df.drop('Attack type', axis=1)
y = df['Attack type']

print(f"Splitting data (Total rows: {len(df)})...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models_config = {
    'DecisionTree': (DecisionTreeClassifier(max_depth=10, random_state=42), 'decision_tree_model.pkl'),
    'RandomForest': (RandomForestClassifier(n_estimators=10, max_depth=10, n_jobs=-1, random_state=42), 'random_forest_model.pkl'),
    'LogisticRegression': (LogisticRegression(max_iter=100, random_state=42), 'logistic_model.pkl'),
    'MLP': (MLPClassifier(hidden_layer_sizes=(50,), max_iter=20, random_state=42), 'mlp_model.pkl'),
    'XGBoost': (XGBClassifier(n_estimators=10, max_depth=5, eval_metric='mlogloss', random_state=42), 'xgboost_model.pkl'),
    'AdaBoost': (AdaBoostClassifier(n_estimators=10, random_state=42), 'adaboost_model.pkl'),
    'Stacking': (StackingClassifier(
        estimators=[
            ('dt', DecisionTreeClassifier(max_depth=5)),
            ('lr', LogisticRegression(max_iter=100))
        ],
        final_estimator=LogisticRegression(),
        n_jobs=-1
    ), 'stacking_model.pkl')
}

for name, (model, path) in models_config.items():
    print(f"Training {name}...")
    try:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"  Accuracy: {acc:.4f}")
        print(f"  Saving to {path}...")
        joblib.dump(model, path)
    except Exception as e:
        print(f"  Failed to train {name}: {e}")

print("All models trained and saved!")
