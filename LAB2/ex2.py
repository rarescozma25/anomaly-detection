import matplotlib.pyplot as plt
from pyod.utils.data import generate_data_clusters
from pyod.models.knn import KNN
from sklearn.metrics import balanced_accuracy_score

# # Using the function generate data clusters generate a 2-dimensional dataset
# with 400 train samples and 200 test samples that are organized in 2 clusters,
# with 0.1 contamination.

X_train, X_test, Y_train, Y_test = generate_data_clusters(
    n_train=400, 
    n_test=200, 
    n_clusters=2, 
    n_features=2, 
    contamination=0.1, 
    random_state=42
)

n_neighors_val = 15


model = KNN(n_neighbors=n_neighors_val)
model.fit(X_train)



y_train_pred = model.labels_
y_test_pred = model.predict(X_test)

balanced_acc_train = balanced_accuracy_score(Y_train, y_train_pred)
balanced_acc_test = balanced_accuracy_score(Y_test, y_test_pred)

print(f"Balanced accuracy score on training data: {balanced_acc_train:.4f}")
print(f"Balanced accuracy score on test data: {balanced_acc_test:.4f}")



fig, axes = plt.subplots(2, 2, figsize=(12, 10))


def plot_data(ax, X, y, title):
    inliers = X[y == 0]
    ax.scatter(inliers[:, 0], inliers[:, 1], c='royalblue', edgecolor='k', s=30, alpha=0.7, label='Inlier (0)')
    outliers = X[y == 1]
    ax.scatter(outliers[:, 0], outliers[:, 1], c='crimson', edgecolor='k', s=50, marker='X', label='Outlier (1)')
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='upper right', fontsize=8)

plot_data(axes[0, 0], X_train, Y_train, "Ground Truth (Train)")


plot_data(axes[0, 1], X_train, y_train_pred, f"Predicted (Train) - Acc: {balanced_acc_train:.2f}")


plot_data(axes[1, 0], X_test, Y_test, "Ground Truth (Test)")


plot_data(axes[1, 1], X_test, y_test_pred, f"Predicted (Test) - Acc: {balanced_acc_test:.2f}")

plt.tight_layout()
plt.show()

