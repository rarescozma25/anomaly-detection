from pyod.utils.data import generate_data
import numpy as np
from sklearn.metrics import balanced_accuracy_score

# Generate a unidimensional dataset with 10 % contamination rate, 1000 training samples and no testing samples using generate data(). Try to detect the
# anomalies in the dataset by using the Z-scores. In order to do that you should
# compute the Z-score threshold that would classify the given percent (contamination rate) of data as anomalies (use np.quantile() function). Compute the
# balanced accuracy of the designed method.

X_train, X_test, y_train, y_test = generate_data(
    n_train=1000,
    n_test=0,
    n_features=1,
    contamination=0.1
)
# |Z| = |(X - mean) / std|
mean = X_train.mean()
std = X_train.std()
z_scores = np.abs((X_train - mean) / std)

threshold = np.quantile(np.abs(z_scores), 0.9)  # 90% of data is normal, 10% is outliers

y_pred = (z_scores > threshold).astype(int)  # 1 for outliers, 0 for normal

b_accuracy = balanced_accuracy_score(y_train, y_pred)

print(f'Z-score threshold: {threshold:.4f}')
print (f'Balanced Accuracy: {b_accuracy:.4f}')
