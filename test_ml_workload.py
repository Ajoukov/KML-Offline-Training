import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from __global_paths import *
import os

with open(TMP_DIR + 'X_test_wl.pkl', 'rb') as f:
    X_test = pickle.load(f)
with open(TMP_DIR + 'y_test_wl.pkl', 'rb') as f:
    y_test = pickle.load(f)

model = Workload()
model.load_state_dict(torch.load(TMP_DIR + "workload.pth"))

model.eval()
with torch.no_grad():
    predicted = model(X_test)

predicted = predicted.numpy()

print(predicted)

predicted = np.argmax(predicted, axis=1)
y_test_np = np.argmax(np.array(y_test), axis=1)

# conf_matrix = confusion_matrix(np.argmax(y_test_np, axis=1), np.argmax(predicted, axis=1))
conf_matrix = confusion_matrix(y_test_np, predicted)

print("Confusion Matrix:")
print(conf_matrix)

# Plot Confusion Matrix using matplotlib
def plot_confusion_matrix(cm, classes, title='Confusion Matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

# Define class labels (replace with your own labels if needed)
class_labels = np.unique(y_test_np)

plt.figure(figsize=(10, 7))
plot_confusion_matrix(conf_matrix, classes=class_labels, title='Confusion Matrix')
plt.savefig(OUT_DIR + 'confusion_matrix_tree.png')
plt.show()

# Classification statistics
accuracy = accuracy_score(y_test_np, predicted)
precision = precision_score(y_test_np, predicted, average='weighted')
recall = recall_score(y_test_np, predicted, average='weighted')
f1 = f1_score(y_test_np, predicted, average='weighted')

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# Save predictions and actuals for further inspection
pd.DataFrame(predicted.astype(int)).to_csv(TST_DIR + "tree_predictions.csv", index=False, header=False)
pd.DataFrame(y_test_np.astype(int)).to_csv(TST_DIR + "tree_actual.csv", index=False, header=False)
