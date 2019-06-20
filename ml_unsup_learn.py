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


# 2. Visualization with hierarchical clustering and t-SNE
# -------------------------------------------------------


# 3. Decorrelating your data and dimension reduction
# --------------------------------------------------


# 4. Discovering interpretable features
# -------------------------------------




