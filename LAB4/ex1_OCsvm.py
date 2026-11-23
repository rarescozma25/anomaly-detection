import numpy as np
import matplotlib.pyplot as plt
from pyod.utils.data import generate_data
from pyod.models.ocsvm import OCSVM
from sklearn.metrics import balanced_accuracy_score, roc_auc_score



X_train, X_test, y_train, y_test = generate_data(
    n_train=300,
    n_test=200,
    n_features=3,
    contamination=0.15,
    random_state=42,
)



# LINEAR KERNEL
model = OCSVM(kernel="linear", contamination=0.15)
model.fit(X_train)


y_test_pred = model.predict(X_test)
y_test_scores = model.decision_function(X_test)


ba = balanced_accuracy_score(y_test, y_test_pred)
auc = roc_auc_score(y_test, y_test_scores)

print(f"Balanced accuracy:{ba:.4f}")
print(f"ROC AUC:{auc:.4f}")

y_train_pred = model.labels_

fig = plt.figure(figsize=(12, 10))

date = [
    (X_train, y_train, "Train - real"),
    (X_train, y_train_pred, "Train - prezis"),
    (X_test, y_test, "Test - real"),
    (X_test, y_test_pred, "Test - prezis"),
]

for i, (X, y, titlu) in enumerate(date):
    ax = fig.add_subplot(2, 2, i + 1, projection="3d")
    ax.scatter(X[y == 0, 0], X[y == 0, 1], X[y == 0, 2], c="blue", label="inlier")
    ax.scatter(X[y == 1, 0], X[y == 1, 1], X[y == 1, 2], c="red", label="outlier")
    ax.set_title(titlu)
    ax.legend()

plt.show()





# RBF KERNEL


model_rbf = OCSVM(kernel="rbf", contamination=0.15)
model_rbf.fit(X_train)

y_train_pred_rbf = model_rbf.labels_
y_test_pred_rbf = model_rbf.predict(X_test)
y_test_scores_rbf = model_rbf.decision_function(X_test)

ba_rbf = balanced_accuracy_score(y_test, y_test_pred_rbf)
auc_rbf = roc_auc_score(y_test, y_test_scores_rbf)

print(f"RBF - Balanced accuracy:{ba_rbf:.4f}")
print(f"RBF - ROC AUC:{auc_rbf:.4f}")

fig = plt.figure(figsize=(12, 10))

date = [
    (X_train, y_train, "Train - real"),
    (X_train, y_train_pred_rbf, "Train - prezis (RBF)"),
    (X_test, y_test, "Test - real"),
    (X_test, y_test_pred_rbf, "Test - prezis (RBF)"),
]

for i, (X, y, titlu) in enumerate(date):
    ax = fig.add_subplot(2, 2, i + 1, projection="3d")
    ax.scatter(X[y == 0, 0], X[y == 0, 1], X[y == 0, 2], c="blue", label="inlier")
    ax.scatter(X[y == 1, 0], X[y == 1, 1], X[y == 1, 2], c="red", label="outlier")
    ax.set_title(titlu)
    ax.legend()

plt.show()