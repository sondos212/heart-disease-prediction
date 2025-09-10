# 06_hyperparameter_tuning.py (with Pipeline)
import os
import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib

# -------------------------------
# 1️⃣ Load data
# -------------------------------
data = pd.read_csv(
    r"C:\Users\sondo\OneDrive - Alexandria National University\Desktop\Sprints final project\heart+disease\heart_disease_cleaned.csv"
)
X = data.drop('target', axis=1)
y = data['target']

# -------------------------------
# 2️⃣ Split data
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------------
# 3️⃣ Define parameter grids with Pipelines
# -------------------------------
param_grids = {
    "Random Forest": {
        "model": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(random_state=42))
        ]),
        "params": {
            "clf__n_estimators": [100, 200, 300],
            "clf__max_depth": [None, 5, 10],
            "clf__min_samples_split": [2, 5, 10],
        },
    },
    "Logistic Regression": {
        "model": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000))
        ]),
        "params": {
            "clf__C": [0.01, 0.1, 1, 10],
            "clf__solver": ["liblinear", "lbfgs"],
        },
    },
    "SVM": {
        "model": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(probability=True))
        ]),
        "params": {
            "clf__C": [0.1, 1, 10],
            "clf__kernel": ["linear", "rbf"],
        },
    },
}

# -------------------------------
# 4️⃣ Run GridSearchCV for each model
# -------------------------------
results = []
best_model = None
best_score = 0

for name, config in param_grids.items():
    print(f" Tuning {name} ...")
    grid = GridSearchCV(
        config["model"], config["params"], cv=5, scoring="accuracy", n_jobs=-1
    )
    grid.fit(X_train, y_train)

    results.append({
        "Model": name,
        "Best Params": grid.best_params_,
        "Best CV Accuracy": grid.best_score_,
    })

    if grid.best_score_ > best_score:
        best_score = grid.best_score_
        best_model = grid.best_estimator_

# -------------------------------
# 5️⃣ Display results
# -------------------------------
results_df = pd.DataFrame(results)
print("\n Hyperparameter Tuning Results:\n")
print(results_df.to_string(index=False))

# -------------------------------
# 6️⃣ Save the best pipeline safely
# -------------------------------
models_dir = r"C:\Users\sondo\OneDrive - Alexandria National University\Desktop\Sprints final project\heart+disease\models"
os.makedirs(models_dir, exist_ok=True)  # ✅ Create folder if not exists

model_path = os.path.join(models_dir, "final_model.pkl")
joblib.dump(best_model, model_path)

print(f"\n Best pipeline saved successfully at: {model_path}")
