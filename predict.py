import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
from PIL import Image


# =====================================
# LOAD TRAINED MODEL
# =====================================

model = load_model(
    "models/mnist_cnn.keras"
)

print("Model loaded successfully!")


# =====================================
# IMAGE PATH
# =====================================

IMAGE_PATH = r"sample_digit.png"


# =====================================
# LOAD IMAGE
# =====================================

img = Image.open(IMAGE_PATH)

print("Original image size:", img.size)


# =====================================
# CONVERT TO GRAYSCALE
# =====================================

img = img.convert("L")


# =====================================
# RESIZE IMAGE
# =====================================

img = img.resize((28, 28))


# =====================================
# CONVERT IMAGE TO NUMPY ARRAY
# =====================================

img_array = np.array(img)


# =====================================
# NORMALIZE PIXEL VALUES
# =====================================

img_array = img_array.astype("float32") / 255.0


# =====================================
# ADD CHANNEL DIMENSION
# =====================================

img_array = np.expand_dims(
    img_array,
    axis=-1
)


# =====================================
# ADD BATCH DIMENSION
# =====================================

img_array = np.expand_dims(
    img_array,
    axis=0
)


print("Input shape:", img_array.shape)


# =====================================
# PREDICTION
# =====================================

predictions = model.predict(
    img_array
)


# =====================================
# GET PREDICTED DIGIT
# =====================================

predicted_digit = np.argmax(
    predictions[0]
)


confidence = np.max(
    predictions[0]
) * 100


# =====================================
# DISPLAY RESULT
# =====================================

print("\nPrediction Result")
print("-------------------")

print(
    f"Predicted Digit : {predicted_digit}"
)

print(
    f"Confidence      : {confidence:.2f}%"
)


# =====================================
# TOP 3 PREDICTIONS
# =====================================

top3_indices = np.argsort(
    predictions[0]
)[::-1][:3]


print("\nTop 3 Predictions")
print("-------------------")

for idx in top3_indices:

    probability = predictions[0][idx] * 100

    print(
        f"Digit {idx} : {probability:.2f}%"
    )


# =====================================
# DISPLAY IMAGE
# =====================================

plt.imshow(
    img_array[0].squeeze(),
    cmap="gray"
)

plt.title(
    f"Predicted Digit: {predicted_digit}"
)

plt.axis("off")

plt.show()