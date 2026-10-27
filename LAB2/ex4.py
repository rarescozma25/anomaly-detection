import numpy as np
import scipy.io as sio
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
from pyod.utils.utility import standardizer
from pyod.models.combination import average, maximization

from pyod.models.knn import KNN

dataset=sio.loadmat('cardio.mat')
X = dataset['X']
Y = dataset['y'].ravel() #ravel flattens the array to 1D

contamination_rate = np.sum(Y) / len(Y) # np.sum(Y) counts the number of anomalies (assuming anomalies are labeled as 1), and len(Y) gives the total number of samples.

X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.4, random_state=42, stratify=Y
)

X_train_norm, X_test_norm = standardizer(X_train, X_test)

neighbors_range= [ 30,  40,  50,  60,  70,  80,  90, 100, 110, 120]
tr_list = []
te_list = []
for n_neighbors in neighbors_range:
    model = KNN(n_neighbors=n_neighbors, contamination=contamination_rate).fit(X_train_norm)
    tr_list.append(model.decision_scores_)
    te_list.append(model.decision_function(X_test_norm))

tr_mat, te_mat = standardizer(np.column_stack(tr_list), np.column_stack(te_list)) # standardize the decision scores for each model

tr_score_avg = average(tr_mat)
te_score_avg = average(te_mat)

threshold_avg = np.quantile(tr_score_avg, 1 - contamination_rate) # threshold for the average combination method
ba_train_avg = balanced_accuracy_score(y_train, (tr_score_avg >= threshold_avg).astype(int))
ba_test_avg = balanced_accuracy_score(y_test, (te_score_avg >= threshold_avg).astype(int))

print("Average combination method:")
print("Balanced accuracy on training data: {:.4f}".format(ba_train_avg))
print("Balanced accuracy on test data: {:.4f}".format(ba_test_avg))

tr_score_max = maximization(tr_mat)
te_score_max = maximization(te_mat)

threshold_max = np.quantile(tr_score_max, 1 - contamination_rate) # threshold for the maximization combination method
ba_train_max = balanced_accuracy_score(y_train, (tr_score_max >= threshold_max).astype(int))
ba_test_max = balanced_accuracy_score(y_test, (te_score_max >= threshold_max).astype(int))

print("\nMaximization combination method:")
print("Balanced accuracy on training data: {:.4f}".format(ba_train_max))
print("Balanced accuracy on test data: {:.4f}".format(ba_test_max))