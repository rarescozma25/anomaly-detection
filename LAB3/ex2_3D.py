import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from pyod.models.iforest import IForest
from pyod.models.dif import DIF
from pyod.models.loda import LODA

X, y = make_blobs(n_samples=1000, centers=[[0, 10, 0], [10, 0, 10]], cluster_std=1.0, random_state=42)

model = IForest(contamination=0.02, random_state=42)

model.fit(X)

#1000 samples
test_data = np.random.uniform(low=-10.0, high=20, size=(1000, 3))



scores_iforest= model.decision_function(test_data)

plt.figure(figsize=(7, 6))
ax_single = plt.subplot(projection='3d')
ax_single.scatter(
    test_data[:, 0], test_data[:, 1], test_data[:, 2], 
    c=scores_iforest, 
    cmap='viridis', 
    edgecolors='k', 
    s=30, 
    linewidth=0.5
)
plt.show()
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


fig = plt.figure(figsize=(18, 5))

axes = [
    fig.add_subplot(1, 3, 1, projection='3d'),
    fig.add_subplot(1, 3, 2, projection='3d'),
    fig.add_subplot(1, 3, 3, projection='3d')
]


sc1 = axes[0].scatter(test_data[:, 0], test_data[:, 1], test_data[:, 2], c=scores_iforest, cmap='viridis', s=20)
axes[0].set_title("Isolation Forest")
fig.colorbar(sc1, ax=axes[0], shrink=0.6)


sc2 = axes[1].scatter(test_data[:, 0], test_data[:, 1], test_data[:, 2], c=scores_dif, cmap='viridis', s=20)
axes[1].set_title("DIF")
fig.colorbar(sc2, ax=axes[1], shrink=0.6)


sc3 = axes[2].scatter(test_data[:, 0], test_data[:, 1], test_data[:, 2], c=scores_loda, cmap='viridis', s=20)
axes[2].set_title("LODA")
fig.colorbar(sc3, ax=axes[2], shrink=0.6)

plt.tight_layout()
plt.show()

# 2.5 
bins_to_test = [5, 30, 100] #Few bins = wide bands (underfitting); many bins = dense grid of lines (overfitting).
fig_loda = plt.figure(figsize=(15, 5))

for idx, n_b in enumerate(bins_to_test):
    loda_custom = LODA(n_bins=n_b, contamination=0.02)
    loda_custom.fit(X)
    scores = loda_custom.decision_function(test_data)
    
    ax = fig_loda.add_subplot(1, len(bins_to_test), idx + 1, projection='3d')
    sc = ax.scatter(test_data[:, 0], test_data[:, 1], test_data[:, 2], c=scores, cmap='viridis', s=20)
    ax.set_title(f"LODA (n_bins = {n_b})")
    fig_loda.colorbar(sc, ax=ax, shrink=0.6)

plt.tight_layout()
plt.show()



dif_values = [5, 20, 50] #Few neurons = simpler/rigid boundaries; 
                         #many neurons = smooth, curved contours around data clusters without false straight-line artifacts
fig_dif = plt.figure(figsize=(15, 5))

for idx, val in enumerate(dif_values):
    dif_custom = DIF(n_ensemble=val, contamination=0.02, random_state=42)
    dif_custom.fit(X)
    scores = dif_custom.decision_function(test_data)
    
    ax = fig_dif.add_subplot(1, len(dif_values), idx + 1, projection='3d')
    sc = ax.scatter(test_data[:, 0], test_data[:, 1], test_data[:, 2], c=scores, cmap='viridis', s=20)
    ax.set_title(f"DIF (n_ensemble = {val})")
    fig_dif.colorbar(sc, ax=ax, shrink=0.6)

plt.tight_layout()
plt.show()