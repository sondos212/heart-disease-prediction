# 01_data_preprocessing.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from ucimlrepo import fetch_ucirepo

# -------------------------------
# 1️⃣ Load Cleveland dataset from UCI Repo
# -------------------------------
heart_disease = fetch_ucirepo(id=45)

# Features and target
X = heart_disease.data.features
y = heart_disease.data.targets

# دمج الـ features مع الـ target في DataFrame واحد
data = pd.concat([X, y], axis=1)

print("Initial data shape:", data.shape)
print("Missing values per column:\n", data.isnull().sum())

# -------------------------------
# 2️⃣ Convert target to binary
# -------------------------------
data['num'] = data['num'].apply(lambda x: 1 if x > 0 else 0)  # العمود اسمه num هنا
data.rename(columns={'num': 'target'}, inplace=True)

# -------------------------------
# 3️⃣ Handle missing values
# -------------------------------
for col in data.columns:
    if data[col].isnull().sum() > 0:
        if data[col].dtype in ['int64','float64']:
            data[col].fillna(data[col].median(), inplace=True)
        else:
            data[col].fillna(data[col].mode()[0], inplace=True)

print("Missing values handled. Any remaining?\n", data.isnull().sum())

# -------------------------------
# 4️⃣ Encode categorical features if needed
# -------------------------------
categorical_cols = ['cp', 'restecg', 'slope', 'thal', 'ca']
data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)

# -------------------------------
# 5️⃣ Standardize numerical features
# -------------------------------
num_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'fbs', 'sex', 'exang']
scaler = StandardScaler()
data[num_cols] = scaler.fit_transform(data[num_cols])

# -------------------------------
# 6️⃣ Save cleaned dataset
# -------------------------------
output_path = r"C:\Users\sondo\OneDrive - Alexandria National University\Desktop\Heart_Disease_Project\heart_disease_cleaned.csv"
data.to_csv(output_path, index=False)

print("Data preprocessing completed. Cleaned dataset saved at:")
print(output_path)
