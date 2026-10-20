from pyod.utils.data import generate_data
import matplotlib.pyplot as plt
from pyod.models.knn import KNN
from sklearn.metrics import ConfusionMatrixDisplay, balanced_accuracy_score, confusion_matrix, roc_curve
# Generare date de antrenare și testare
X_train, X_test, y_train, y_test = generate_data(
    n_train=400,
    n_test=100,
    n_features=2,
    contamination=0.1
)

normal_samples = y_train == 0
outlier_samples = y_train == 1

plt.scatter(
    X_train[normal_samples, 0],
    X_train[normal_samples, 1],
    color='steelblue',
    label='Normal',
    edgecolor='k'
)
plt.scatter(
    X_train[outlier_samples, 0],
    X_train[outlier_samples, 1],
    color='crimson',
    label='Outlier',
    edgecolor='k'
)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()



model=KNN(contamination=0.1)
model.fit(X_train)
predictions_test=model.predict(X_test)

# predictions_train=model.predict(X_train)
y_test_scores = model.decision_function(X_test)

confusion = confusion_matrix(y_test, predictions_test)
ConfusionMatrixDisplay(
    confusion_matrix=confusion,
    display_labels=['Normal', 'Outlier']
).plot(cmap='Blues')
plt.title('Matricea de confuzie')
plt.show()

# confusion_train = confusion_matrix(y_train, predictions_train)
# ConfusionMatrixDisplay(
#     confusion_matrix=confusion_train,
#     display_labels=['Normal', 'Outlier']
# ).plot(cmap='Blues')
# plt.title('Matricea de confuzie - Date de antrenare')
# plt.show()

tn, fp, fn, tp = confusion_matrix(y_test, predictions_test).ravel()
b_acc=balanced_accuracy_score(y_test, predictions_test)
print(f'True Negatives: {tn}, False Positives: {fp}, False Negatives: {fn}, True Positives: {tp}')
print(f'Balanced Accuracy: {b_acc}')


# contam_rates=[0.05, 0.1, 0.15, 0.2]
# for rate in contam_rates:
#     model=KNN(contamination=rate)
#     model.fit(X_train)
#     predictions_test=model.predict(X_test)
#     predictions_train=model.predict(X_train)
#     y_test_scores = model.decision_function(X_test)

#     confusion = confusion_matrix(y_test, predictions_test)
#     ConfusionMatrixDisplay(
#         confusion_matrix=confusion,
#         display_labels=['Normal', 'Outlier']
#     ).plot(cmap='Blues')
#     plt.title(f'Matricea de confuzie - Rata de contaminare: {rate}')
#     plt.show()
#     tn, fp, fn, tp = confusion_matrix(y_test, predictions_test).ravel()
#     b_acc=balanced_accuracy_score(y_test, predictions_test)
#     print(f'Rata de contaminare: {rate} - True Negatives: {tn}, False Positives: {fp}, False Negatives: {fn}, True Positives: {tp}')
#     print(f'Rata de contaminare: {rate} - Balanced Accuracy: {b_acc}')



fpr, tpr, thresholds = roc_curve(y_test, y_test_scores)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color="darkorange", lw=2, label="KNN ROC Curve")
plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random Guess")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC)")
plt.legend(loc="lower right")
plt.grid(True)
plt.show()


