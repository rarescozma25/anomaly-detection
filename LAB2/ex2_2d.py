import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# y = a * x1 + b * x2 + c + epsilon
a = 2.0
b = -1.5
c_intercept = 1.0
n_samples = 50
n_special = 15  #for special points in each category

def compute_leverage_scores(X1, X2):
    # 1 for intercept term, plus x1 and x2
    X_design = np.column_stack((np.ones(X1.shape[0]), X1, X2))
    
    # Compute the hat matrix H = X * (X^T * X)^(-1) * X^T
    H = X_design @ np.linalg.inv(X_design.T @ X_design) @ X_design.T
    
    # Compute leverage scores as diagonal elements of H
    return np.diag(H)

mu = 0.0
sigma = np.sqrt(0.2)

#regular data
x1_reg = np.random.uniform(-3, 3, n_samples)
x2_reg = np.random.uniform(-3, 3, n_samples)
noise_reg = np.random.normal(mu, sigma, n_samples)
y_reg = a * x1_reg + b * x2_reg + c_intercept + noise_reg

# High variance for x1 and x2
x1_high_x = np.random.uniform(8, 10, n_special)
x2_high_x = np.random.uniform(8, 10, n_special)
y_high_x = a * x1_high_x + b * x2_high_x + c_intercept + np.random.normal(mu, sigma, n_special)

# High variance for y
x1_high_y = np.random.uniform(-0.5, 0.5, n_special)
x2_high_y = np.random.uniform(-0.5, 0.5, n_special)
y_high_y = a * x1_high_y + b * x2_high_y + c_intercept + 15.0 + np.random.normal(mu, sigma, n_special)

# High variance for both
x1_high_xy = np.random.uniform(8, 10, n_special)
x2_high_xy = np.random.uniform(8, 10, n_special)
y_high_xy = a * x1_high_xy + b * x2_high_xy + c_intercept + 15.0 + np.random.normal(mu, sigma, n_special)


X1_all = np.concatenate([x1_reg, x1_high_x, x1_high_y, x1_high_xy])
X2_all = np.concatenate([x2_reg, x2_high_x, x2_high_y, x2_high_xy])
Y_all = np.concatenate([y_reg, y_high_x, y_high_y, y_high_xy])


lev_scores = compute_leverage_scores(X1_all, X2_all)


fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(projection='3d')


ax.scatter(x1_reg, x2_reg, y_reg, color='blue', alpha=0.5, s=40, label='Regular')
ax.scatter(x1_high_x, x2_high_x, y_high_x, color='green', alpha=0.8, s=50, label='High var X')
ax.scatter(x1_high_y, x2_high_y, y_high_y, color='orange', alpha=0.8, s=50, label='High var Y')
ax.scatter(x1_high_xy, x2_high_xy, y_high_xy, color='purple', alpha=0.8, s=50, label='High var X & Y')


special_indices_groups = [
    (np.mean(x1_high_x), np.mean(x2_high_x), np.mean(y_high_x), n_samples, 'High X'),
    (np.mean(x1_high_y), np.mean(x2_high_y), np.mean(y_high_y), n_samples + n_special, 'High Y'),
    (np.mean(x1_high_xy), np.mean(x2_high_xy), np.mean(y_high_xy), n_samples + 2 * n_special, 'High X&Y')
]

for mean_x1, mean_x2, mean_y, pt_idx, txt in special_indices_groups:
    ax.text(mean_x1, mean_x2, mean_y + 1, f"{txt}\nh={lev_scores[pt_idx]:.2f}", 
            ha='center', fontsize=8, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.8))

ax.set_title("Leverage Scores in 2D Linear Regression", fontsize=12, fontweight='bold')
ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.set_zlabel("y")
ax.legend(loc='upper left', fontsize=9)

plt.tight_layout()
plt.show()