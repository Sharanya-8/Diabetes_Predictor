import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# --- 1. Load and Prepare Data ---
try:
    
    df = pd.read_csv('diabetes.csv') 
except FileNotFoundError:
    print("ERROR: diabetes.csv not found! Please place the dataset file in the same directory.")
    exit()

cols_to_replace = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in cols_to_replace:
    
    non_zero_mean = df[df[col] != 0][col].mean()
    df[col] = df[col].replace(0, non_zero_mean)

X = df.drop('Outcome', axis=1) 
y = df['Outcome']              

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# --- 2. Scaling (Essential for Logistic Regression and KNN) ---

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


joblib.dump(scaler, 'models/scaler.pkl')

# --- 3. Train Models, Evaluate, and Save ---

models = {
    'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
    'LogisticRegression': LogisticRegression(solver='liblinear', random_state=42),
    'KNeighbors': KNeighborsClassifier(n_neighbors=11) # A common choice for this dataset
}

results = {}

for name, model in models.items():
    print(f"\n--- Training {name} ---")
    
    model.fit(X_train_scaled, y_train) 
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1] 
    
    # Evaluation Metrics
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred),
        'Trained_Model': model 
    }
    results[name] = metrics
    
    joblib.dump(model, f'models/{name.lower().replace(" ", "")}_model.pkl')
    print(f"✅ {name} model saved as models/{name.lower().replace(' ', '')}_model.pkl")

# --- 4. Display Final Results ---
print("\n" + "="*50)
print("             MODEL EVALUATION METRICS")
print("="*50)

for name, metrics in results.items():
    print(f"\nModel: {name}")
    print(f"  Accuracy:  {metrics['Accuracy']:.4f}")
    print(f"  Precision: {metrics['Precision']:.4f}")
    print(f"  Recall:    {metrics['Recall']:.4f}")
    print(f"  F1-Score:  {metrics['F1-Score']:.4f}")

print("\nAll models and the scaler have been successfully trained and saved!")