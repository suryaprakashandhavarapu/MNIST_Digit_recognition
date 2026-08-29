import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import mnist

from sklearn.metrics import (
    classification_report,
    confusion_matrix
)


# =====================================
# LOAD MODEL
# =====================================

model = load_model(
    "models/mnist_cnn.keras"
)

print("Model loaded successfully!")


# =====================================
# LOAD MNIST TEST DATA
# =====================================

(_, _), (X_test, y_test) = mnist.load_data()

print("\nTest images:", X_test.shape)
print("Test labels:", y_test.shape)


# =====================================
# PREPROCESS TEST DATA
# =====================================

# Convert pixel values from 0-255 to 0-1

X_test = X_test.astype("float32") / 255.0


# Add channel dimension

X_test = np.expand_dims(
    X_test,
    axis=-1
)

print("After preprocessing:", X_test.shape)


# =====================================
# PREDICTIONS
# =====================================

predictions = model.predict(
    X_test,
    batch_size=64
)

y_pred = np.argmax(
    predictions,
    axis=1
)


# =====================================
# TEST ACCURACY
# =====================================

accuracy = np.mean(
    y_test == y_pred
)

print("\nTest Accuracy")
print("-------------------")

print(
    f"Accuracy : {accuracy * 100:.2f}%"
)


# =====================================
# CLASSIFICATION REPORT
# =====================================

print("\nClassification Report")
print("-------------------")

print(
    classification_report(
        y_test,
        y_pred
    )
)


# =====================================
# CONFUSION MATRIX
# =====================================

cm = confusion_matrix(
    y_test,
    y_pred
)


# =====================================
# DISPLAY CONFUSION MATRIX
# =====================================

plt.figure(
    figsize=(8, 6)
)

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=range(10),
    yticklabels=range(10)
)

plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")

plt.title(
    "MNIST Confusion Matrix"
)

plt.tight_layout()


# =====================================
# SAVE CONFUSION MATRIX
# =====================================

plt.savefig(
    "results/confusion_matrix.png"
)

plt.show()