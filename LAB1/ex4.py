from pyod.utils.data import generate_data
import numpy as np
from sklearn.metrics import balanced_accuracy_score

# Same as Ex. 3 but for a multidimensional dataset. Choose your own mean
# vector µ and covariance matrix Σ and build your dataset by hand starting with
# the samples generated from a multivariate standard normal distribution, x ∼
# N (0, I). Using the Cholesky decomposition of the covariance matrix Σ = LLT
# draw your samples y like this: y = Lx + µ and then compute the Z-scores. All
# other tasks as in Ex. 3.


mu = np.array([2, 3])  # Mean vector
cov = np.array([[1, 0.5], [0.5, 1]])  # Covariance matrix
n_samples = 1000
contamination_rate = 0.1

n_outliers = 100  # 100 anomalies
n_inliers = 900  # 900 normal samples

# generate normal samples, I is the identity matrix
np.random.seed(42)
x_inliers = np.random.randn(n_inliers, 2)  # Standard normal samples for inliers

# Descompunerea Cholesky: cov = L * L^T, used for transforming standard normal samples to the desired distribution
L = np.linalg.cholesky(cov) 

# Transformarea datelor: y = Lx + mu, @ is the matrix multiplication operator in Python, .T is the transpose operator
y_inliers = x_inliers @ L.T + mu #these are the normal samples, transformed to have the desired mean and covariance

# generate outlier samples uniformly in a larger range
y_outliers = np.random.uniform(low=-6, high=10, size=(n_outliers, 2)) #these are the outlier samples, generated uniformly in a larger range to ensure they are far from the normal samples

# Combine the inliers and outliers into a single dataset
y = np.vstack([y_inliers, y_outliers])
y_true = np.hstack([np.zeros(n_inliers), np.ones(n_outliers)])  # 0 normal, 1 outlier

#computed inv_l because we need to transform the data back to standard normal space 
# to compute the Z-scores. The Z-score is computed as the distance from the mean in standard deviations, and for multivariate data, this involves transforming the data using the inverse of the Cholesky factor L.
inv_L = np.linalg.inv(L)
x_transformed = (y - mu) @ inv_L.T
z_scores = np.linalg.norm(x_transformed, axis=1)

# Compute the threshold for the given contamination rate
threshold = np.quantile(z_scores, 1 - contamination_rate)  # 90% of data is normal, 10% is outliers

y_pred = (z_scores > threshold).astype(int)  # 1 for outliers, 0 for normal

# Balanced Accuracy
b_accuracy = balanced_accuracy_score(y_true, y_pred)

print(f'Z-score threshold: {threshold:.4f}')
print(f'Balanced Accuracy: {b_accuracy:.4f}')