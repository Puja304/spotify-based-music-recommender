import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
import hdbscan
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os
os.environ['LOKY_MAX_CPU_COUNT'] = '4'

df = pd.read_csv("Merged-Puja.csv", header=0, parse_dates=['date'])

#get all relevant features
audio_features = ['popularity', 'danceability', 'energy','key','loudness','mode','speechiness','acousticness','instrumentalness','liveness','valence','tempo']
genre_features = [col for col in df.columns if col.startswith('genre_')]

X = df[audio_features + genre_features]

# cluster_pipeline = Pipeline([
#     ('scaler', StandardScaler()),
#     ('hdbscan', hdbscan.HDBSCAN(min_cluster_size=8))
# ])

#can't use pipeline due to conditional logic ahead
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

dbscan = DBSCAN(eps=4, min_samples=20)
db_labels = dbscan.fit_predict(X_scaled)
df['cluster'] = db_labels



#big issue: too many songs going to the noise category or specific ones
#reclustering the noise:

next_cluster_id = 1000
final_labels = df['cluster'].copy()

large_clusters = df['cluster'].value_counts()
large_clusters = large_clusters[large_clusters > int((df.shape[0]/5))].index.tolist()  #anything more than 10%

for cluster_id in large_clusters:
    mask = df['cluster'] == cluster_id
    X_subset = X_scaled[mask]

    n_kmeans = 5  # or set dynamically based on size
    kmeans = KMeans(n_clusters=n_kmeans, random_state=42)
    new_labels = kmeans.fit_predict(X_subset)

    subset_indices = df[mask].index  # indices of songs in this large cluster

    for i in range(n_kmeans):
        selected_indices = subset_indices[new_labels == i]
        final_labels.loc[selected_indices] = next_cluster_id
        next_cluster_id += 1



df['cluster'] = final_labels

#drop tiny clusters 
cluster_counts = df['cluster'].value_counts()
valid_clusters = cluster_counts[cluster_counts >= 10].index
df = df[df['cluster'].isin(valid_clusters)]

#save
df.to_csv("Puja-clustered.csv", index=False)

#descrive clusters
cluster_sizes = df['cluster'].value_counts().sort_index()
print("\nCluster Sizes:")
for cluster_id, size in cluster_sizes.items():
    label = f"Noise (-1)" if cluster_id == -1 else f"Cluster {cluster_id}"
    print(f"{label}: {size} songs")
#visualize data
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(StandardScaler().fit_transform(X))  

# Create scatter plot
plt.figure(figsize=(10, 6))

# Slice X_reduced to only include retained rows
X_reduced_filtered = X_reduced[df.index]

plt.scatter(X_reduced_filtered[:, 0], X_reduced_filtered[:, 1],
            c=df['cluster'], cmap='tab10', s=30, alpha=0.8)

plt.title("HDBSCAN + KMeans Clustering of Songs (2D PCA Projection)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.colorbar(label='Cluster')
plt.grid(True)
plt.tight_layout()
plt.show()
