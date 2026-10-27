import matplotlib.pyplot as plt
from pyod.utils.data import generate_data_clusters
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from sklearn.datasets import make_blobs
from sklearn.metrics import balanced_accuracy_score


# First, generate
# 2 clusters (200 and 100 samples respectively) with 2-dimensional samples using
# (-10,-10) and (10, 10) as centers, 2 and 6 as standard deviations using
# sklearn.datasets.make blobs() function.
X, y = make_blobs(
    n_samples=[200, 100], 
    centers=[(-10, -10), (10, 10)], 
    cluster_std=[2, 6], 
    n_features=2, 
    random_state=42
)

knn_model = KNN(contamination=0.07)
knn_model.fit(X)

y_pred_knn = knn_model.labels_

lof_model = LOF(contamination=0.07)
lof_model.fit(X)
y_pred_lof = lof_model.labels_


fig, axes = plt.subplots(1, 2, figsize=(14, 6))


axes[0].scatter(X[y_pred_knn == 0, 0], X[y_pred_knn == 0, 1], c='blue', s=25, label='Inliers')
axes[0].scatter(X[y_pred_knn == 1, 0], X[y_pred_knn == 1, 1], c='red', s=35, label='Outliers')
axes[0].set_title('KNN')
axes[0].grid(True, linestyle=':', alpha=0.6)


axes[1].scatter(X[y_pred_lof == 0, 0], X[y_pred_lof == 0, 1], c='blue', s=25, label='Inliers')
axes[1].scatter(X[y_pred_lof == 1, 0], X[y_pred_lof == 1, 1], c='red', s=35, label='Outliers')
axes[1].set_title('LOF')
axes[1].grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
plt.show()