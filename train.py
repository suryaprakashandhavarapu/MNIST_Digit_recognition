import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout,
    BatchNormalization
)
from tensorflow.keras.callbacks import EarlyStopping


# =====================================
# LOAD MNIST DATASET
# =====================================

(X_train, y_train), (X_test, y_test) = mnist.load_data()

print("Training images:", X_train.shape)
print("Training labels:", y_train.shape)

print("Testing images:", X_test.shape)
print("Testing labels:", y_test.shape)


# =====================================
# PREPROCESSING
# =====================================

# Convert pixel values from 0-255 to 0-1
X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0


# CNN expects:
# (samples, height, width, channels)

X_train = np.expand_dims(X_train, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)


print("After preprocessing:")
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)


# =====================================
# CNN MODEL
# =====================================

model = Sequential([

    Conv2D(
        32,
        (3, 3),
        activation="relu",
        padding="same",
        input_shape=(28, 28, 1)
    ),

    BatchNormalization(),

    MaxPooling2D((2, 2)),


    Conv2D(
        64,
        (3, 3),
        activation="relu",
        padding="same"
    ),

    BatchNormalization(),

    MaxPooling2D((2, 2)),


    Flatten(),

    Dense(128, activation="relu"),

    Dropout(0.3),

    Dense(10, activation="softmax")
])


# =====================================
# MODEL SUMMARY
# =====================================

model.summary()


# =====================================
# COMPILE MODEL
# =====================================

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# =====================================
# EARLY STOPPING
# =====================================

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)


# =====================================
# TRAIN MODEL
# =====================================

history = model.fit(
    X_train,
    y_train,
    validation_split=0.1,
    epochs=20,
    batch_size=64,
    callbacks=[early_stop]
)


# =====================================
# SAVE MODEL
# =====================================

model.save("models/mnist_cnn.keras")

print("\nModel saved successfully!")


# =====================================
# TRAINING ACCURACY GRAPH
# =====================================

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training vs Validation Accuracy")
plt.legend()

plt.savefig(
    "results/training_accuracy.png"
)

plt.show()


# =====================================
# TRAINING LOSS GRAPH
# =====================================

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()

plt.savefig(
    "results/training_loss.png"
)

plt.show()