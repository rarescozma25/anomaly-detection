import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from pyod.models.iforest import IForest
from pyod.models.dif import DIF
from pyod.models.loda import LODA

X, y = make_blobs(n_samples=1000, centers=[[10, 0], [0, 10]], cluster_std=1.0, random_state=42)

model = IForest(contamination=0.02, random_state=42)

model.fit(X)

#1000 samples
test_data = np.random.uniform(low=-10.0, high=20, size=(1000, 2))



scores_iforest= model.decision_function(test_data)

plt.figure(figsize=(7, 6))
plt.scatter(
    test_data[:, 0], test_data[:, 1], 
    c=scores_iforest, 
    cmap='viridis', 
    edgecolors='k', 
    s=30, 
    linewidth=0.5
)
# we can see paralel lines in the plot, which are artifacts of the isolation forest algorithm. 
# These lines represent the decision boundaries created by the isolation trees in the forest. 

# plt.colorbar(label='Anomaly Score')
# plt.title("Isolation Forest Artifacts (Uniform Test Data)")
# plt.xlabel("Feature 1")
# plt.ylabel("Feature 2")
# plt.grid(True, linestyle="--", alpha=0.5)
# plt.show()

dif_model = DIF(contamination=0.02, random_state=42)
dif_model.fit(X)
scores_dif = dif_model.decision_function(test_data)

loda_model = LODA(contamination=0.02, random_state=42)
loda_model.fit(X)
scores_loda = loda_model.decision_function(test_data)


fig, axes = plt.subplots(1, 3, figsize=(18, 5))


sc1 = axes[0].scatter(test_data[:, 0], test_data[:, 1], c=scores_iforest, cmap='viridis', s=20)
axes[0].set_title("Isolation Forest")
fig.colorbar(sc1, ax=axes[0])


sc2 = axes[1].scatter(test_data[:, 0], test_data[:, 1], c=scores_dif, cmap='viridis', s=20)
axes[1].set_title("DIF")
fig.colorbar(sc2, ax=axes[1])


sc3 = axes[2].scatter(test_data[:, 0], test_data[:, 1], c=scores_loda, cmap='viridis', s=20)
axes[2].set_title("LODA")
fig.colorbar(sc3, ax=axes[2])

plt.tight_layout()
plt.show()

# 2.5 
bins_to_test = [5, 30, 100] #Few bins = wide bands (underfitting); many bins = dense grid of lines (overfitting).
fig_loda, axes_loda = plt.subplots(1, len(bins_to_test), figsize=(15, 5))

for idx, n_b in enumerate(bins_to_test):
    loda_custom = LODA(n_bins=n_b, contamination=0.02)
    loda_custom.fit(X)
    scores = loda_custom.decision_function(test_data)
    
    sc = axes_loda[idx].scatter(test_data[:, 0], test_data[:, 1], c=scores, cmap='viridis', s=20)
    axes_loda[idx].set_title(f"LODA (n_bins = {n_b})")
    fig_loda.colorbar(sc, ax=axes_loda[idx])

plt.tight_layout()
plt.show()



dif_values = [5, 20, 50] #Few neurons = simpler/rigid boundaries; 
                         #many neurons = smooth, curved contours around data clusters without false straight-line artifacts
fig_dif, axes_dif = plt.subplots(1, len(dif_values), figsize=(15, 5))

for idx, val in enumerate(dif_values):
    dif_custom = DIF(n_ensemble=val, contamination=0.02, random_state=42)
    dif_custom.fit(X)
    scores = dif_custom.decision_function(test_data)
    
    sc = axes_dif[idx].scatter(test_data[:, 0], test_data[:, 1], c=scores, cmap='viridis', s=20)
    axes_dif[idx].set_title(f"DIF (n_ensemble = {val})")
    fig_dif.colorbar(sc, ax=axes_dif[idx])

plt.tight_layout()
plt.show()
