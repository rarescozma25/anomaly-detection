import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from pyod.utils.utility import standardizer
from pyod.models.ocsvm import OCSVM
from pyod.models.deep_svdd import DeepSVDD

data = loadmat("shuttle.mat")
X = data["X"]
y = data["y"].ravel()

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.5,
    random_state=42,
    stratify=y,
)

# standardizer
X_train, X_test = standardizer(X_train, X_test)

contamination = y_train.mean()
print("Contamination:", contamination)

# OCSVM
# ocsvm = OCSVM(kernel="rbf", contamination=contamination)
# ocsvm.fit(X_train)

# y_pred = ocsvm.predict(X_test)
# y_scores = ocsvm.decision_function(X_test)

# print(f"OCSVM - Balanced accuracy: {balanced_accuracy_score(y_test, y_pred):.4f}")
# print(f"OCSVM - ROC AUC: {roc_auc_score(y_test, y_scores):.4f}")

# DeepSVDD
# deep = DeepSVDD(n_features=X_train.shape[1], contamination=contamination, random_state=42)
# deep.fit(X_train)

# y_pred = deep.predict(X_test)
# y_scores = deep.decision_function(X_test)

# print(f"DeepSVDD - Balanced accuracy: {balanced_accuracy_score(y_test, y_pred):.4f}")
# print(f"DeepSVDD - ROC AUC: {roc_auc_score(y_test, y_scores):.4f}")


architectures = [
    [8, 4],
    [16, 8],
    [32, 16],
    [64, 32],
    [64, 32, 16],
    [128, 64, 32],
]

results = []

for arh in architectures:
    deep = DeepSVDD(
        n_features=X_train.shape[1],
        hidden_neurons=arh,
        contamination=contamination,
        random_state=42,
    )
    deep.fit(X_train)

    y_pred = deep.predict(X_test)
    y_scores = deep.decision_function(X_test)

    ba = balanced_accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_scores)

    results.append((arh, ba, auc))


for arh, ba, auc in results:
    print(f"DeepSVDD {arh} - BA: {ba:.4f}, ROC AUC: {auc:.4f}")