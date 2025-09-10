# 05_unsupervised_learning.py
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score

# -------------------------------
# 1️⃣ Load cleaned data
# -------------------------------
data = pd.read_csv(
    r"C:\Users\sondo\OneDrive - Alexandria National University\Desktop\Sprints final project\heart+disease\heart_disease_cleaned.csv"
)
X = data.drop('target', axis=1)
y = data['target']

# -------------------------------
# 2️⃣ Elbow Method for KMeans
# -------------------------------
inertia = []
sil_scores = []
K_range = range(2, 11)  # نبدأ من 2 لأن cluster=1 مش منطقي للتقييم

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)
    sil_scores.append(silhouette_score(X, kmeans.labels_))

plt.figure(figsize=(12,5))

# Elbow curve
plt.subplot(1,2,1)
plt.plot(K_range, inertia, marker='o')
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia")
plt.title("Elbow Method for KMeans")

# Silhouette scores
plt.subplot(1,2,2)
plt.plot(K_range, sil_scores, marker='o', color='orange')
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Analysis for KMeans")

plt.tight_layout()
plt.show()

# -------------------------------
# 3️⃣ Apply KMeans with k=2 (binary disease vs no disease)
# -------------------------------
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
clusters_k = kmeans.fit_predict(X)

# Reduce to 2D for visualization using PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters_k, cmap="viridis", alpha=0.7)
plt.title("K-Means Clustering (PCA-reduced)")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.show()

print(" K-Means Adjusted Rand Index (vs true labels):", adjusted_rand_score(y, clusters_k))

# -------------------------------
# 4️⃣ Hierarchical Clustering
# -------------------------------
linked = linkage(X, method='ward')

plt.figure(figsize=(12, 6))
dendrogram(linked, truncate_mode="level", p=5)
plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Samples")
plt.ylabel("Distance")
plt.show()

# Fit Agglomerative Clustering
agg = AgglomerativeClustering(n_clusters=2, linkage="ward")
clusters_h = agg.fit_predict(X)

print(" Hierarchical Adjusted Rand Index (vs true labels):", adjusted_rand_score(y, clusters_h))
