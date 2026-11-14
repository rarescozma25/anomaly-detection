import scipy.io
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pyod.models.iforest import IForest
from pyod.models.dif import DIF
from pyod.models.loda import LODA
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

mat = scipy.io.loadmat('shuttle.mat')
X = mat['X']
y = mat['y'].ravel()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


iforest = IForest(contamination=0.07, random_state=42).fit(X_train_scaled)
dif = DIF(contamination=0.07, random_state=42).fit(X_train_scaled)
loda = LODA(contamination=0.07).fit(X_train_scaled)


scores_iforest = iforest.decision_function(X_test_scaled)
preds_iforest = iforest.predict(X_test_scaled)

scores_dif = dif.decision_function(X_test_scaled)
preds_dif = dif.predict(X_test_scaled)

scores_loda = loda.decision_function(X_test_scaled)
preds_loda = loda.predict(X_test_scaled)


print("--- Isolation Forest ---")
print("Balanced Accuracy:", balanced_accuracy_score(y_test, preds_iforest))
print("ROC AUC:", roc_auc_score(y_test, scores_iforest))

print("\n--- DIF ---")
print("Balanced Accuracy:", balanced_accuracy_score(y_test, preds_dif))
print("ROC AUC:", roc_auc_score(y_test, scores_dif))

print("\n--- LODA ---")
print("Balanced Accuracy:", balanced_accuracy_score(y_test, preds_loda))
print("ROC AUC:", roc_auc_score(y_test, scores_loda))


# CONCLUZII:
# 1. Isolation Forest: Scoruri excelente (ROC AUC ~0.996) si acuratete ridicata (~0.97), cel mai stabil model pentru date tabulare.
# 2. DIF: Performanta buna la ordonare (ROC AUC ~0.96), dar acuratete moderata (~0.79) din cauza pragului de contaminare.
# 3. LODA: Dificultati majore pe date multidimensionale (ROC AUC ~0.56); proiecțiile 1D nu surprind complexitatea anomaliilor.
