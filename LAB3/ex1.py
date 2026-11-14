import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs


#generate a  2D dataset that follows a standard normal distribution

X, y = make_blobs(n_samples=500, centers=[[0, 0]], cluster_std=1.0, random_state=42)

# plt.figure(figsize=(6, 6))
# plt.scatter(X[:, 0], X[:, 1], alpha=0.7, edgecolors='k', s=30)
# plt.title("500 Points Generated from N(0, I)")
# plt.xlabel("Feature 1")
# plt.ylabel("Feature 2")
# plt.grid(True, linestyle="--", alpha=0.5)
# plt.show()



W = np.random.multivariate_normal(
    mean=[0, 0],
    cov=np.eye(2), #np.eye(2) creates a 2x2 identity matrix, which serves as the covariance matrix for the multivariate normal distribution. This means that the two features are uncorrelated and have a variance of 1.
    size=5
)

W = W / np.linalg.norm(W, axis=1, keepdims=True) #because unit-length projection vectors

projected_data = X @ W.T #project the data onto the 5 random vectors

# #  For each histogram compute the probability corresponding to each
# bin and use them to compute the anomaly score of a sample as the mean
# of the probabilities (corresponding to each histogram).

n_bins = 30
probab_matrix = np.zeros(np.shape(projected_data))

for i in range(5):
    proj_vals = projected_data[:, i]
    
    
    hist_range = (proj_vals.min() - 1.0, proj_vals.max() + 1.0) #we need to extend the range of the histogram to ensure that all projected values are included within the bins. By subtracting 1.0 from the minimum value and adding 1.0 to the maximum value, we create a buffer around the data points, which helps to avoid edge cases where some projected values might fall outside the defined histogram range. This ensures that all data points are accounted for in the histogram calculation.
    
    # the np.histogram function computes the histogram of the projected values. It returns two arrays: counts, which contains the number of projected values that fall into each bin, and bin_edges, which defines the edges of the bins. The bins parameter specifies the number of bins to use, and the range parameter defines the range of values to consider for the histogram.
    counts, bin_edges = np.histogram(proj_vals, bins=n_bins, range=hist_range)
    
    # probabilities for each bin are calculated by dividing the counts of projected values in each bin by the total number of projected values (len(X)). This gives us the relative frequency of projected values in each bin, which can be interpreted as the probability of a projected value falling into that bin.
    bin_probs = counts / len(X)
    
    bin_indices = np.digitize(proj_vals, bin_edges[:-1]) - 1 # find the indices of the bins to which each projected value belongs. The np.digitize function returns the indices of the bins for each projected value, and we subtract 1 to convert the indices to a 0-based index system (since np.digitize returns 1-based indices).
    bin_indices = np.clip(bin_indices, 0, n_bins - 1) # ensures that the bin indices are within the valid range of 0 to n_bins - 1. This is important because some projected values might fall outside the defined histogram range, and we want to avoid indexing errors when accessing the bin probabilities.
    
    probab_matrix[:, i] = bin_probs[bin_indices] # assigns the probabilities corresponding to each projected value's bin to the probab_matrix. For each projected value, we look up its bin index in bin_indices and use that index to retrieve the corresponding probability from bin_probs. This way, probab_matrix contains the probabilities for each projected value across all 5 random vectors.

mean_probabilities = np.mean(probab_matrix, axis=1)

print("Scores for the first 5 samples:", mean_probabilities[:5])

X_test = np.random.uniform(low=-3.0, high=3.0, size=(500, 2))

test_projected = X_test @ W.T 

test_probab_matrix = np.zeros(np.shape(test_projected))

for i in range(5):
    proj_vals = test_projected[:, i]
    hist_range = (proj_vals.min() - 1.0, proj_vals.max() + 1.0)
    counts, bin_edges = np.histogram(proj_vals, bins=n_bins, range=hist_range)
    
    bin_probs = counts / len(X_test)
    
    bin_indices = np.digitize(proj_vals, bin_edges[:-1]) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    
    test_probab_matrix[:, i] = bin_probs[bin_indices]
    
test_anomaly_scores = -np.mean(test_probab_matrix, axis=1)

plt.scatter(X_test[:, 0], X_test[:, 1], c=test_anomaly_scores, cmap='viridis')
plt.colorbar()
plt.show()