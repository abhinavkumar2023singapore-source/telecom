# Install if needed: !pip install pandas scikit-learn xgboost imbalanced-learn matplotlib seaborn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# 1. Load data (replace with your Kaggle paths)
train = pd.read_csv('train.csv')  # Adjust path
test = pd.read_csv('test.csv')

print("Train shape:", train.shape)
print("Test shape:", test.shape)
print("\nChurn distribution:\n", train['Churn'].value_counts(normalize=True))  # Check imbalance [web:23]

# 2. EDA Quick view
print("\nMissing values:\n", train.isnull().sum())
print("\nData types:\n", train.dtypes)

# Visualize churn by key features (adjust column names as needed)
fig, axes = plt.subplots(1, 3, figsize=(15,5))
sns.countplot(data=train, x='Contract', hue='Churn', ax=axes[0])  # Common high-impact feature [web:23][web:39]
sns.histplot(data=train, x='tenure', hue='Churn', ax=axes[1], bins=20)
sns.boxplot(data=train, x='Churn', y='MonthlyCharges', ax=axes[2])
plt.tight_layout()
# plt.show()

# 3. Preprocessing function
def preprocess(df):
    df = df.copy()
    # Drop useless
    df.drop(['customerID'], axis=1, inplace=True, errors='ignore')

    # Handle TotalCharges (often string with spaces)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)

    # Encode binary Yes/No to 0/1
    for col in ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']:
        if col in df.columns:
            df[col] = df[col].map({'Yes':1, 'No':0})

    # Label encode other categoricals
    cat_cols = df.select_dtypes('object').columns.drop(['Churn'], errors='ignore')
    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col].astype(str))

    return df

train_processed = preprocess(train)
# Select only the features used in the dashboard
selected_features = ['tenure', 'MonthlyCharges', 'Contract']
X = train_processed[selected_features]
y = train_processed['Churn']

# 4. Train-test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale numeric (tenure, MonthlyCharges)
scaler = StandardScaler()
num_cols = ['tenure', 'MonthlyCharges']
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_val[num_cols] = scaler.transform(X_val[num_cols])

# 5. Skip SMOTE for small dataset
X_train_bal, y_train_bal = X_train, y_train

# 6. Models - Baseline to Advanced
models = {
    'Logistic': LogisticRegression(random_state=42),
    'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
    'XGBoost': xgb.XGBClassifier(random_state=42, eval_metric='logloss')
}

results = {}
for name, model in models.items():
    model.fit(X_train_bal, y_train_bal)
    y_pred_proba = model.predict_proba(X_val)[:,1]
    auc = roc_auc_score(y_val, y_pred_proba)
    results[name] = auc
    print(f"{name} ROC-AUC: {auc:.4f}")

# Best model
best_model = models[max(results, key=results.get)]
print("\nBest model:", max(results, key=results.get))

# Detailed metrics
y_pred = best_model.predict(X_val)
print("\nClassification Report:\n", classification_report(y_val, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_val, y_pred))

# Feature importance (for trees/boosting)
if hasattr(best_model, 'feature_importances_'):
    feat_imp = pd.DataFrame({'feature': X.columns, 'importance': best_model.feature_importances_}).sort_values('importance', ascending=False)
    print("\nTop Features:\n", feat_imp.head())
    sns.barplot(data=feat_imp.head(10), x='importance', y='feature')
    plt.title('Top 10 Feature Importances')
    # plt.show()  # Tenure, MonthlyCharges, Contract usually top [web:30][web:39]

# 7. Cross-validation
scores = cross_val_score(best_model, X_train_bal, y_train_bal, cv=5, scoring='roc_auc')
print(f"CV ROC-AUC: {scores.mean():.4f} (+/- {scores.std()*2:.4f})")

# 8. Predict on test & save submission
test_processed = preprocess(test)
test_processed[num_cols] = scaler.transform(test_processed[num_cols])
test_preds = best_model.predict_proba(test_processed)[:,1]
submission = pd.DataFrame({'id': test.index, 'Churn': test_preds})  # Adjust id column
submission.to_csv('submission.csv', index=False)
print("Submission saved!")

# Save model and scaler for Flask app
import joblib
joblib.dump(best_model, 'telecom_model.pkl')
joblib.dump(scaler, 'telecom_scaler.pkl')
print("Model and scaler saved!")
