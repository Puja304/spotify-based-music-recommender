import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt
from scipy.stats import f_oneway
from scipy.stats import spearmanr
import math
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
import hdbscan
from sklearn.decomposition import PCA
import os
os.environ['LOKY_MAX_CPU_COUNT'] = '4'

plt.close('all')


#create a new column that concatenates temp and icon so it's a new feature
df = pd.read_csv("Merged-Puja.csv", header=0, parse_dates=['date'])

def classify(temp):
    if temp < 0:
        return "very-cold"
    elif temp < 10:
        return "cold"
    elif temp < 16:
        return "normal"
    elif temp < 22:
        return "warm"
    else:
        return "hot"

df['temp-bin'] = df['temp'].apply(classify)
df['temp-icon'] = df['temp-bin'] + df['icon']


song_features = ['popularity','danceability','energy','key','loudness','mode','speechiness','acousticness','instrumentalness','liveness','valence','tempo','time_signature']
cols = 3
rows = math.ceil(len(song_features) / cols)
fig, axes = plt.subplots(rows, cols, figsize=(15, 10 * rows))
axes = axes.flatten()  # so you can index with a single loop

for i, feature in enumerate(song_features):
    ax = axes[i]
    sns.boxplot(data=df, x='temp-icon', y=feature, ax=ax)
    ax.set_title(f"{feature.capitalize()} by Temp-Icon")
    ax.tick_params(axis='x', rotation=45)

# Hide any unused subplots
for j in range(len(song_features), rows * cols):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()
plt.savefig("Puja-boxplots.png")
plt.close()
# anova
print("\n ANOVA Test Results:")

anova_results = []
for feature in song_features:
    groups = [group[feature].dropna().values for name, group in df.groupby('temp-icon')]
    f_stat, p_val = f_oneway(*groups)
    anova_results.append((feature, f_stat, p_val))
    print(f"{feature}: F = {f_stat:.2f}, p = {p_val:.5f}")

# rank
print("\nTop 3 weather-sensitive features:")
has_mode = False
top_features = sorted(anova_results, key=lambda x: -x[1])[:3]
for feat, f_stat, p in top_features:
    print(f"{feat} (F = {f_stat:.2f}, p = {p:.5f})")
    df["ft_"+feat] = df[feat]
    if(feat == 'mode') :
        has_mode = True

# heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(df[song_features].corr(), annot=True, cmap='coolwarm')
plt.title("\nSong Feature Correlations")
plt.tight_layout()
plt.savefig("heatmap-Puja.png")
plt.close()

df['temp-bin-num'] = df['temp-bin'].map({'very-cold': 0, 'cold': 1, 'normal': 2, 'warm': 3, 'hot': 4})

for feature in song_features:
    corr, p = spearmanr(df['temp-bin-num'], df[feature])
    print(f"{feature}: Spearman ρ = {corr:.2f}, p = {p:.5f}")




genre_features = [col for col in df.columns if col.startswith('genre_')]
audio_features = [col for col in df.columns if col.startswith('tf')]
if (not has_mode):
    audio_features.append('mode')

X = df[audio_features + genre_features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

dbscan = DBSCAN(eps=4, min_samples=20)
db_labels = dbscan.fit_predict(X_scaled)
df['cluster'] = db_labels
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

#drop tiny clusters `       `
cluster_counts = df['cluster'].value_counts()
valid_clusters = cluster_counts[cluster_counts >= 10].index
df = df[df['cluster'].isin(valid_clusters)]

#save
df.to_csv("Puja-feature-clustered.csv", index=False)

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
