import numpy as np
from sklearn.cluster import KMeans


# This agent takes in the set of all positions, and breaks them up into small clusters.
# From there, it manually solves the travelling salesman problem. Then it applies the algorithm again on the clusters.
# Or for the simpler approach, it can just connect them all at the origin point.
# This reduces the cost from n! to x * (n / x)!
def generate_clusters(points, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(points)
    labels = kmeans.predict(points)

    # Create a dictionary to store points for each cluster
    clusters = {i: [] for i in range(n_clusters)}

    # Assign points to clusters
    for i, label in enumerate(labels):
        clusters[label].append(points[i])

    # Convert clusters to NumPy arrays
    for key in clusters:
        clusters[key] = np.array(clusters[key])

    return clusters


