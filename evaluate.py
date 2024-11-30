import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Load the results
utk_true_age = np.load('UTKFace_true_age.npy', allow_pickle=True)
utk_pred_age = np.load('UTKFace_pred_age.npy', allow_pickle=True)
utk_true_gender = np.load('UTKFace_true_gender.npy', allow_pickle=True)
utk_pred_gender = np.load('UTKFace_pred_gender.npy', allow_pickle=True)

own_true_age = np.load('OwnFace_true_age.npy', allow_pickle=True)
own_pred_age = np.load('OwnFace_pred_age.npy', allow_pickle=True)
own_true_gender = np.load('OwnFace_true_gender.npy', allow_pickle=True)
own_pred_gender = np.load('OwnFace_pred_gender.npy', allow_pickle=True)

# Plotting Age Predictions
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.scatter(utk_true_age, utk_pred_age, alpha=0.5)
plt.plot([0, 100], [0, 100], 'r--')  # Ideal prediction line
plt.title('UTKFace Dataset - Age Prediction')
plt.xlabel('True Age')
plt.ylabel('Predicted Age')

plt.subplot(1, 2, 2)
plt.scatter(own_true_age, own_pred_age, alpha=0.5)
plt.plot([0, 100], [0, 100], 'r--')  # Ideal prediction line
plt.title('Own Dataset - Age Prediction')
plt.xlabel('True Age')
plt.ylabel('Predicted Age')

plt.tight_layout()
plt.show()

# Plotting Gender Predictions (confusion matrices)
def plot_confusion_matrix(true_labels, pred_labels, title):
    cm = confusion_matrix(true_labels, pred_labels)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

plot_confusion_matrix(utk_true_gender, utk_pred_gender, 'UTKFace Dataset - Gender Prediction')
plot_confusion_matrix(own_true_gender, own_pred_gender, 'Own Dataset - Gender Prediction')
