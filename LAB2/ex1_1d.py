import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# y = a * x + b + epsilon
a = 2.0
b = 1.0
n_samples = 50
n_special = 15  #for special points in each category

def compute_leverage_scores(X):
    # 1 for intercept term, plus x
    X_design = np.hstack((np.ones((X.shape[0], 1)), X.reshape(-1, 1)))
    
    # Compute the hat matrix H
    H = X_design @ np.linalg.inv(X_design.T @ X_design) @ X_design.T
    
    # Compute leverage scores as diagonal elements of H
    return np.diag(H)


mu = 0.0
sigma = np.sqrt(0.2)

# 1. Date Regular
x_reg = np.random.uniform(-3, 3, n_samples)
noise_reg = np.random.normal(mu, sigma, n_samples)
y_reg = a * x_reg + b + noise_reg

x_high_x = np.random.uniform(9, 11, n_special)
y_high_x = a * x_high_x + b + np.random.normal(mu, sigma, n_special)

x_high_y = np.random.uniform(-0.5, 0.5, n_special)
y_high_y = a * x_high_y + b + 15.0 + np.random.normal(mu, sigma, n_special)

x_high_xy = np.random.uniform(9, 11, n_special)
y_high_xy = a * x_high_xy + b + 15.0 + np.random.normal(mu, sigma, n_special)


X_all = np.concatenate([x_reg, x_high_x, x_high_y, x_high_xy])
Y_all = np.concatenate([y_reg, y_high_x, y_high_y, y_high_xy])

lev_scores = compute_leverage_scores(X_all)


plt.figure(figsize=(10, 6))

x_line = np.linspace(-11, 11, 100)
plt.plot(x_line, a * x_line + b, 'r--', label='Model True ($y=2x+1$)', alpha=0.7)


plt.scatter(x_reg, y_reg, color='blue', alpha=0.5, marker='o', s=40, label='Regular')
plt.scatter(x_high_x, y_high_x, color='green', alpha=0.8, marker='o', s=50, label='High var X')
plt.scatter(x_high_y, y_high_y, color='orange', alpha=0.8, marker='o', s=50, label='High var Y')
plt.scatter(x_high_xy, y_high_xy, color='purple', alpha=0.8, marker='o', s=50, label='High var X & Y')


special_indices_groups = [
    (np.mean(x_high_x), np.mean(y_high_x), n_samples, 'High X'),
    (np.mean(x_high_y), np.mean(y_high_y), n_samples + n_special, 'High Y'),
    (np.mean(x_high_xy), np.mean(y_high_xy), n_samples + 2 * n_special, 'High X&Y')
]

for mean_x, mean_y, pt_idx, txt in special_indices_groups:
    plt.annotate(f"{txt}\nh={lev_scores[pt_idx]:.2f}", 
                (mean_x, mean_y),
                textcoords="offset points", xytext=(0, 15), 
                ha='center', fontsize=9, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.8))

plt.title("Leverage Scores in 1D Linear Regression", fontsize=12, fontweight='bold')
plt.xlabel("x")
plt.ylabel("y")
plt.legend(loc='upper left', fontsize=9)
plt.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
plt.show()