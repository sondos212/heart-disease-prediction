# 04_supervised_learning.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, auc, roc_auc_score
)
from sklearn.feature_selection import RFE
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
# 2️⃣ Select Top 10 Features using RFE
# -------------------------------
log_reg = LogisticRegression(max_iter=1000, solver="liblinear")
rfe = RFE(log_reg, n_features_to_select=10)
rfe.fit(X, y)

selected_features = X.columns[rfe.support_]
X = X[selected_features]

print(" Top 10 features selected for training:")
print(list(selected_features))
print()

# -------------------------------
# 3️⃣ Split data
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------------
# 4️⃣ Define models
# -------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "SVM": SVC(probability=True, random_state=42)
}

# -------------------------------
# 5️⃣ Train, Evaluate & Collect Results
# -------------------------------
results = []

plt.figure(figsize=(8,6))  # ROC figure
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:,1]  # for ROC

    # Collect metrics
    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred),
        "AUC Score": roc_auc_score(y_test, y_prob)
    })

    # Plot ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.plot(fpr, tpr, label=f"{name} (AUC = {auc(fpr, tpr):.2f})")

# -------------------------------
# 6️⃣ Display Results in a Table
# -------------------------------
results_df = pd.DataFrame(results)
print("\n Model Performance Comparison:\n")
print(results_df.to_string(index=False))

# -------------------------------
# 7️⃣ Finalize ROC Curve Plot
# -------------------------------
plt.plot([0,1], [0,1], 'k--')  # random guess line
plt.title("ROC Curve Comparison")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.show()
