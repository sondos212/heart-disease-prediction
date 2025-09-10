# 03_feature_selection.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# -------------------------------
# 1️⃣ Load cleaned data
# -------------------------------
data = pd.read_csv(
    r"C:\Users\sondo\OneDrive - Alexandria National University\Desktop\Sprints final project\heart+disease\heart_disease_cleaned.csv"
)

X = data.drop('target', axis=1)
y = data['target']

# -------------------------------
# 2️⃣ Feature importance with Random Forest
# -------------------------------
rf = RandomForestClassifier(random_state=42)
rf.fit(X, y)

importances = pd.Series(rf.feature_importances_, index=X.columns)
importances_sorted = importances.sort_values(ascending=False)

plt.figure(figsize=(12,6))
importances_sorted.plot(kind='bar')
plt.title('Feature Importance - Random Forest')
plt.ylabel('Importance Score')
plt.xticks(rotation=90)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# -------------------------------
# 3️⃣ Recursive Feature Elimination (RFE)
# -------------------------------
log_reg = LogisticRegression(max_iter=1000, solver='liblinear')  # solver مناسب للـ RFE
rfe = RFE(log_reg, n_features_to_select=10)
rfe.fit(X, y)

selected_features = X.columns[rfe.support_]

print("\n Top features selected by RFE:")
for f in selected_features:
    print("-", f)

# -------------------------------
# 4️⃣ Chi-Square Test
# -------------------------------
# لازم نعمل Scaling للـ features بحيث تكون قيم موجبة (chi2 requirement)
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

chi_scores, p_values = chi2(X_scaled, y)

chi2_results = pd.DataFrame({
    "Feature": X.columns,
    "Chi2 Score": chi_scores,
    "p-value": p_values
}).sort_values(by="Chi2 Score", ascending=False)

print("\n Chi-Square Test Results:")
print(chi2_results)

plt.figure(figsize=(12,6))
plt.bar(chi2_results["Feature"], chi2_results["Chi2 Score"])
plt.title("Chi-Square Test - Feature Significance")
plt.ylabel("Chi2 Score")
plt.xticks(rotation=90)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
