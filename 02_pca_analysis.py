# 02_pca_analysis.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# -------------------------------
# 1️⃣ Load cleaned data
# -------------------------------
data = pd.read_csv(
    r"C:\Users\sondo\OneDrive - Alexandria National University\Desktop\Sprints final project\heart+disease\heart_disease_cleaned.csv"
)

X = data.drop('target', axis=1)
y = data['target']

# -------------------------------
# 2️⃣ Apply PCA
# -------------------------------
pca = PCA()
X_pca = pca.fit_transform(X)

# -------------------------------
# 3️⃣ Explained variance plot
# -------------------------------
explained_variance = np.cumsum(pca.explained_variance_ratio_)
plt.figure(figsize=(8,5))
plt.plot(range(1, len(explained_variance) + 1), explained_variance, marker='o')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('PCA Explained Variance')
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# -------------------------------
# 4️⃣ PCA with 95% variance
# -------------------------------
pca_opt = PCA(n_components=0.95)  # keep enough components to explain 95% variance
X_pca_opt = pca_opt.fit_transform(X)

print("PCA completed. Number of components retained:", X_pca_opt.shape[1])
