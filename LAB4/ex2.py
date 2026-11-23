import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

data = loadmat("cardio.mat")
X = data["X"]
y = data["y"].ravel()

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    train_size=0.4,
    random_state=42,
    stratify=y,
)

contamination = y_train.mean()

# OneClassSVM uses +1 for inlier and -1 for outlier
y_train_svm = 1 - 2 * y_train
y_test_svm = 1 - 2 * y_test

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", OneClassSVM()),
])

param_grid = {
    "svm__kernel": ["linear", "rbf", "sigmoid", "poly"],
    "svm__gamma": ["scale", "auto", 0.01, 0.1],
    "svm__nu": [0.05, contamination, 0.2, 0.3, 0.5],
}

grid = GridSearchCV(pipe, param_grid, scoring="balanced_accuracy", cv=5)
grid.fit(X_train, y_train_svm)

print("Best params:", grid.best_params_)
print(f"Balanced accuracy (CV): {grid.best_score_:.4f}")

best = grid.best_estimator_
y_test_pred = best.predict(X_test)
y_test_scores = best.decision_function(X_test)

print(f"Balanced accuracy (test): {balanced_accuracy_score(y_test_svm, y_test_pred):.4f}")
print(f"ROC AUC (test): {roc_auc_score(y_test_svm, y_test_scores):.4f}")