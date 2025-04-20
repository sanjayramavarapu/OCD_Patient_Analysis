# Install required packages if not already installed
# pip install xgboost imbalanced-learn pandas scikit-learn matplotlib seaborn

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from imblearn.combine import SMOTETomek
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

# Load dataset
df = pd.read_csv("OCD.csv")

# Create severity label
df['Total YBOCS Score'] = df['Y-BOCS Score (Obsessions)'] + df['Y-BOCS Score (Compulsions)']
def label_severity(score):
    if score <= 15:
        return 'Mild'
    elif 16 <= score <= 25:
        return 'Moderate'
    else:
        return 'Severe'
df['Severity'] = df['Total YBOCS Score'].apply(label_severity)

# Drop unnecessary columns
df.drop(columns=['Patient ID', 'OCD Diagnosis Date',
                 'Y-BOCS Score (Obsessions)', 'Y-BOCS Score (Compulsions)', 'Total YBOCS Score'], inplace=True)

# Encode categorical features
categorical_cols = df.select_dtypes(include='object').columns.tolist()
categorical_cols.remove('Severity')
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Encode target
target_encoder = LabelEncoder()
df['Severity'] = target_encoder.fit_transform(df['Severity'])

# Split the dataset
X = df.drop(columns='Severity')
y = df['Severity']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Apply SMOTE + Tomek to handle class imbalance
smt = SMOTETomek(random_state=42)
X_resampled, y_resampled = smt.fit_resample(X_train, y_train)

# Define XGBoost model with broader grid for better class separation
xgb = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)

param_grid = {
    'n_estimators': [100],
    'max_depth': [6, 8],
    'learning_rate': [0.1, 0.2],
    'subsample': [0.9],
    'colsample_bytree': [0.8, 1.0]
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid_search = GridSearchCV(estimator=xgb,
                           param_grid=param_grid,
                           scoring='f1_macro',
                           cv=cv,
                           n_jobs=-1,
                           verbose=0)

grid_search.fit(X_resampled, y_resampled)

# Evaluate best model
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# Scores
print("Best Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n",
      classification_report(y_test, y_pred, target_names=target_encoder.classes_, zero_division=0))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu',
            xticklabels=target_encoder.classes_,
            yticklabels=target_encoder.classes_)
plt.title("Confusion Matrix - Balanced XGBoost")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()
