# Unsupervised Learning in Python
# -------------------------------
import pandas as pd
import numpy as np

# 1. Clustering for dataset exploration
# -------------------------------------
# Unsupervised learning encompasses a variety of techniques in machine learning, from clustering to
# dimension reduction to matrix factorization
# idea: classify stocks to outperform / underperform and then perform ML techniques instead of traditional ic-analysis

# k-Means..
# ---
from sklearn.cluster import KMeans
# create a KMeans instance with 3 clusters..
# random dataset of 2 vectors..
x = np.array(np.random.normal(size=100)).T # help(np.array)
y = np.array(np.random.normal(size=100)).T
points = pd.DataFrame({'x':x, 'y':y})
model = KMeans(n_clusters=3)
# fit model to points..
model.fit(points)
# determine the cluster labels of new_points..
new_points = pd.DataFrame({'x':np.array(np.random.normal(size=10)).T, 'y':np.array(np.random.normal(size=10)).T})
labels = model.predict(new_points)
print(labels)

# measuring clustering quality..
# ---
# - insertia measures cluster quality: measures how spread out the clusters are (lower is better)
# - distance from each sample to centroid of its cluster
# - choose an "ellbow" in the inertia plot (where inertia begins to decrease more slowly)
import matplotlib.pyplot as plt
xs = new_points['x']
ys = new_points['y']
# make a scatter plot of xs and ys, using labels to define the colors..
plt.scatter(xs,ys,c=labels,alpha=0.5)
# assign the cluster centers..
centroids = model.cluster_centers_
# assign the columns of centroids & plot them..
centroids_x = centroids[:,0]
centroids_y = centroids[:,1]
plt.scatter(centroids_x,centroids_y,marker='D',s=50)
# .. count the number of times each label coincides with estimated: pd.crosstab(df['varieties'],df['lables'])

# transforming features for better clusters
# ---
# StandardScaler() transforms each feature to have mean 0 & variance 1
# Normalizer() rescales each sample (eg. each company's stock price) independently of the other

# 2. Visualization with hierarchical clustering and t-SNE
# -------------------------------------------------------
# t-SNE & hierarchical clustering
# hierarchical clustering (agglomerative):
# 1. every element begins in a separate cluster
# 2. at each step, 2 closest clusters are merged
# 3. continue until all elements in a single cluster
# .. divisive clustering works the other way around
samples = pd.read_csv('data/ml_seeds.csv', header=None)

# hierarchical clustering..
# ---
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt

# calculate the linkage..
mergings = linkage(samples, method='complete')
samples.shape
varieties = ['Kama wheat'] *70 + ['Canadian wheat'] *70 + ['Rosa wheat'] *70
len(varieties)
# plot the dendrogram, using varieties as labels..
dendrogram(mergings,
           labels=varieties,
           leaf_rotation=90,
           leaf_font_size=6,
)

# height of dendogram: distance between merging clusters (eg. cluster with only element 1,2 had distance of 6)
# complete linkage: the distance between clusters is the distance between the furthest points of the clusters
# single linkage: the distance between clusters is the distance between the closest points of the clusters

# another example of hierarchical clustering..
movements_tmp = pd.read_csv('data/ml_company-stock-movements-2010-2015-incl.csv')
companies = movements_tmp.iloc[:,0].values
movements = np.array(movements_tmp)[:,1:]
from sklearn.preprocessing import normalize
# normalize..
normalized_movements = normalize(movements)
# calculate the linkage..
mergings = linkage(normalized_movements, method='complete')
# dendogram..
dendrogram(mergings, labels=companies, leaf_rotation=90, leaf_font_size=6)

# extracting cluster labels..
# ---
from scipy.cluster.hierarchy import fcluster
labels = fcluster(mergings, 6, criterion='distance')
# create a DF with labels and varieties as columns..
df = pd.DataFrame({'labels': labels, 'varieties': varieties})
# create crosstab..
ct = pd.crosstab(df['labels'],df['varieties'])
print(ct)


# t-SNE (for 2-dimensional maps)
# ---
# t-SNE: t-distributed stochastic neighbor embedding
# - maps samples to 2D (or 3D) from higher dimensions & good for inspecting datasets
# - provides great visualizations when the individual samples can be labeled
# - learning rate parameter: try values between 50 and 200
# - tSNE features are different every times but the relative position stay the same

from sklearn.manifold import TSNE
# create a TSNE instance..
model = TSNE(learning_rate=200)
# apply fit_transform to samples..
tsne_features = model.fit_transform(samples)
# select the 0th feature..
xs = tsne_features[:,0]
# select the 1st feature..
ys = tsne_features[:,1]
# scatter plot, coloring by variety_numbers
plt.scatter(xs,ys,c=labels)

# another example with stock-market data..
model = TSNE(learning_rate=50)
tsne_features = model.fit_transform(normalized_movements)
# select the features
xs = tsne_features[:,0]
ys = tsne_features[:,1]
# plot & annotate points..
plt.scatter(xs,ys,alpha=0.5)
for x, y, company in zip(xs, ys, companies):
    plt.annotate(company, (x, y), fontsize=5, alpha=0.75)


# 3. Decorrelating your data and dimension reduction
# --------------------------------------------------
# PCA: fundamental dimension reduction technique
# 1. step: de-correlation
# 2. step: reducing dimension
# pca in scikit learn has 2 methods for pca: fit() learns the transformation from given data & transform() applies
# the learned transformation

# intrinsic dimension: number of features needed to approximate the dataset


# csr_matrix: remembers only non-zero entries (saves space)

# 4. Discovering interpretable features
# -------------------------------------
# NMF: 'Non-negative matrix factorization': expresses samples as combinations of interpretable parts
# - dimension reduction technique
# - unlike PCA, NMF models are interpretable
# - achieves interpretability by decomposing features as sum of their parts
# - in NMF, all sample features must be non-negative!
# - sample reconstruction: multiply components by feature values & add up (can also be expressed as a
#   product of matrices)

# .. For example, it expresses documents as combinations of topics, and images in terms of commonly occurring
# visual patterns. You'll also learn to use NMF to build recommender systems that can find you similar articles
# to read, or musical artists that match your listening history

# case study
# ---
# recommend articles similar to articles being read by customer




















