import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_moons, make_circles, make_classification, make_s_curve, make_swiss_roll
from sklearn.cluster import KMeans, SpectralClustering, DBSCAN, AgglomerativeClustering

# 生成6种不同的数据集
datasets = [
    make_blobs(n_samples=500, centers=3, random_state=1),
    make_moons(n_samples=500, noise=0.05, random_state=1),
    make_circles(n_samples=500, noise=0.05, factor=0.5, random_state=1),
    make_classification(n_samples=500, n_features=2, n_informative=2, n_redundant=0, random_state=1),
    make_s_curve(n_samples=500, random_state=1),
    make_swiss_roll(n_samples=500, random_state=1)
]

# 设置不同的聚类算法和参数
algorithms = [
    ('KMeans', KMeans(n_clusters=3, random_state=1)),
    ('SpectralClustering', SpectralClustering(n_clusters=3, random_state=1)),
    ('DBSCAN', DBSCAN(eps=0.3, min_samples=10)),
    ('AgglomerativeClustering', AgglomerativeClustering(n_clusters=3)),
]

# 绘制每个数据集的聚类效果
figure, axes = plt.subplots(nrows=len(datasets), ncols=len(algorithms), figsize=(15, 20))

for i, dataset in enumerate(datasets):
    X, y = dataset

    for j, (name, algorithm) in enumerate(algorithms):
        algorithm.fit(X)
        labels = algorithm.labels_

        axes[i, j].scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
        axes[i, j].set_title(name)

plt.show()
