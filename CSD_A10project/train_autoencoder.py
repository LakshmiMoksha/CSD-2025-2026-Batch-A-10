# -*- coding: utf-8 -*-
"""
Train Autoencoder (Deep MLP) Only
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("TRAINING AUTOENCODER MODEL")
print("="*60)

print("\n[ STEP 1 ] Loading WSN-DS.csv dataset...")
try:
    df = pd.read_csv('WSN-DS.csv')
except FileNotFoundError:
    print("Error: WSN-DS.csv not found!")
    exit(1)

# Label Encoding
# Label Encoding for Features (if any)
encoders = {}
for col in df.columns:
    if col != 'Attack type' and df[col].dtype == 'object':
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

print("  ✓ Encoders created for feature columns:", list(encoders.keys()))

# Explicit Mapping for Target (Attack type) to match app.py
label_map = {
    'Normal': 0,
    'Grayhole': 1,
    'Blackhole': 2,
    'TDMA': 3,
    'Flooding': 4
}
print("  ✓ Applying expliciting mapping for Attack type:", label_map)
# Map and drop unmapped/NaN if any
df['Attack type'] = df['Attack type'].map(label_map)
df = df.dropna(subset=['Attack type'])

X = df.drop('Attack type', axis=1)
y = df['Attack type']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print(f"OK - Loaded {X.shape[0]} samples with {X.shape[1]} features")

print("\n[ STEP 2 ] Training Autoencoder (Deep MLP)...")
print("  Architecture: Input -> 32 -> 16 -> 10 (Bottleneck) -> 16 -> 32 -> Output")
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
    verbose=True 
)

autoenc.fit(X_train, y_train)
acc = autoenc.score(X_test, y_test)
print(f"\n  Final Accuracy: {acc:.2%}")

print("\n[ STEP 3 ] Saving Model and Encoders...")
joblib.dump(autoenc, 'autoencoder_model.pkl')
joblib.dump(encoders, 'encoders.pkl')
print("  ✓ Saved autoencoder_model.pkl")
print("  ✓ Saved encoders.pkl")

print("\n" + "="*60)
print("SUCCESS - AUTOENCODER TRAINED!")
print("="*60)
