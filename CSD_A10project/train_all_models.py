# -*- coding: utf-8 -*-
"""
Ultra-Simplified Model Training - Sequential Execution
Handles memory constraints gracefully
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, StackingClassifier
from sklearn.neural_network import MLPClassifier  
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("SEQUENTIAL MODEL TRAINER")
print("="*60)

print("\n[ STEP 1 ] Loading WSN-DS.csv dataset...")
df = pd.read_csv('WSN-DS.csv')

# Label Encoding
le = LabelEncoder()
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = le.fit_transform(df[col])

X = df.drop('Attack type', axis=1)
y = df['Attack type']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print(f"OK - Loaded {X.shape[0]} samples with {X.shape[1]} features")

print("\n[ STEP 2 ] Training Basic Models...")

# 1. Decision Tree
print("\n  [1/8] Decision Tree...", end=' ')
dt = DecisionTreeClassifier(random_state=42, max_depth=15)
dt.fit(X_train, y_train)
acc = dt.score(X_test, y_test)
joblib.dump(dt, 'decision_tree_model.pkl')
print(f"OK ({acc:.2%})")

# 2. Random Forest
print("  [2/8] Random Forest...", end=' ')
rf = RandomForestClassifier(n_estimators=80, random_state=42, max_depth=12, n_jobs=2)
rf.fit(X_train, y_train)
acc = rf.score(X_test, y_test)
joblib.dump(rf, 'random_forest_model.pkl')
print(f"OK ({acc:.2%})")

# 3. MLP
print("  [3/8] MLP Neural Network...", end=' ')
mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=100, random_state=42, early_stopping=True, verbose=False)
mlp.fit(X_train, y_train)
acc = mlp.score(X_test, y_test)
joblib.dump(mlp, 'mlp_model.pkl')
print(f"OK ({acc:.2%})")

# 4. Logistic Regression
print("  [4/8] Logistic Regression...", end=' ')
lr = LogisticRegression(max_iter=200, random_state=42, solver='lbfgs', multi_class='multinomial', n_jobs=2)
lr.fit(X_train, y_train)
acc = lr.score(X_test, y_test)
joblib.dump(lr, 'logistic_model.pkl')
print(f"OK ({acc:.2%})")

# 5. XGBoost
print("  [5/8] XGBoost...", end=' ')
xgb = XGBClassifier(n_estimators=80, random_state=42, max_depth=8, learning_rate=0.1, use_label_encoder=False, eval_metric='mlogloss', nthread=2)
xgb.fit(X_train, y_train)
acc = xgb.score(X_test, y_test)
joblib.dump(xgb, 'xgboost_model.pkl')
print(f"OK ({acc:.2%})")

# 6. AdaBoost
print("  [6/8] AdaBoost...", end=' ')
ada = AdaBoostClassifier(n_estimators=40, random_state=42, learning_rate=0.8, algorithm='SAMME')
ada.fit(X_train, y_train)
acc = ada.score(X_test, y_test)
joblib.dump(ada, 'adaboost_model.pkl')
print(f"OK ({acc:.2%})")

# 7. Stacking (lightweight)
print("  [7/8] Stacking Classifier...", end=' ')
stack = StackingClassifier(
    estimators=[
        ('dt', DecisionTreeClassifier(random_state=42, max_depth=8)),
        ('rf', RandomForestClassifier(n_estimators=30, random_state=42, max_depth=8, n_jobs=1))
    ],
    final_estimator=LogisticRegression(max_iter=100, random_state=42),
    cv=2,
    n_jobs=1
)
stack.fit(X_train, y_train)
acc = stack.score(X_test, y_test)
joblib.dump(stack, 'stacking_model.pkl')
print(f"OK ({acc:.2%})")

# 8. Autoencoder (Deep MLP)
print("  [8/8] Autoencoder (Deep MLP)...", end=' ')
autoenc = MLPClassifier(
    hidden_layer_sizes=(32, 16, 10, 16, 32),
    activation='relu',
    solver='adam',
    alpha=0.0001,
    batch_size=256,
    learning_rate='adaptive',
    max_iter=100,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.1,
    verbose=False
)
autoenc.fit(X_train, y_train)
acc = autoenc.score(X_test, y_test)
joblib.dump(autoenc, 'autoencoder_model.pkl')
print(f"OK ({acc:.2%})")

print("\n" + "="*60)
print("SUCCESS - ALL 8 MODELS TRAINED!")
print("="*60)
print("\nYou can now run: python app.py")
