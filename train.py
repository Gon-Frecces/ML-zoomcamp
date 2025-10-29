# -*- coding: utf-8 -*-
"""course_lead_scoring_cleaned.ipynb"""

# ===== 1. Import Dependencies =====
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
import pickle
from tqdm.auto import tqdm

# ===== 2. Load Dataset =====
df = pd.read_csv('course_lead_scoring.csv')

# ===== 3. Split Data =====
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=42)

# ===== 4. Handle Missing Values =====
df_train = df_train.fillna(0)
df_val = df_val.fillna(0)
df_test = df_test.fillna(0)
df_full_train = df_full_train.fillna(0)

# ===== 5. Identify Feature Types =====
categorical = list(df_train.select_dtypes(include='object').columns)
numerical = list(df_train.select_dtypes(include=['int64', 'float64']).columns)
if 'converted' in numerical:
    numerical.remove('converted')  # remove target from features

# ===== 6. Train Function =====
def train(df_train, y_train, C=1.0):
    dicts = df_train[categorical + numerical].to_dict(orient='records')
    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(dicts)

    model = LogisticRegression(C=C, max_iter=2000, solver='lbfgs')
    model.fit(X_train, y_train)

    return dv, model

# ===== 7. Predict Function =====
def predict(df, dv, model):
    dicts = df[categorical + numerical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = model.predict_proba(X_val)[:, 1]
    return y_pred

# ===== 8. K-Fold Cross Validation =====
n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=1)

print("Performing K-Fold Validation:")
for C in [0.001, 0.01, 0.1, 1, 5, 10]:
    scores = []
    for train_idx, val_idx in tqdm(kfold.split(df_full_train), total=n_splits):
        df_train = df_full_train.iloc[train_idx]
        df_val = df_full_train.iloc[val_idx]

        y_train = df_train.converted.values
        y_val = df_val.converted.values

        dv, model = train(df_train, y_train, C=C)
        y_pred = predict(df_val, dv, model)

        auc = roc_auc_score(y_val, y_pred)
        scores.append(auc)

    print(f"C={C:<5}  AUC: {np.mean(scores):.3f} ± {np.std(scores):.3f}")

# ===== 9. Final Model =====
print("\nTraining final model...")
dv, model = train(df_full_train, df_full_train.converted.values, C=1.0)

y_pred = predict(df_test, dv, model)
y_test = df_test.converted.values

auc = roc_auc_score(y_test, y_pred)
print(f"Final Test AUC: {auc:.3f}")

# ===== 10. Save Model =====
output_file = 'lead_scoring_model.bin'
with open(output_file, 'wb') as f_out:
    pickle.dump((dv, model), f_out)

print(f"Model saved to {output_file}")
