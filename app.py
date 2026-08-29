import streamlit as st
import tensorflow as tf
import numpy as np

from tensorflow.keras.models import load_model
from PIL import Image


# =====================================
# PAGE CONFIGURATION
# =====================================

st.set_page_config(
    page_title="MNIST Digit Recognizer",
    page_icon="🔢",
    layout="centered"
)


# =====================================
# TITLE
# =====================================

st.title("🔢 MNIST Handwritten Digit Recognizer")

st.write(
    "Draw a handwritten digit and let the CNN predict it."
)


# =====================================
# LOAD MODEL
# =====================================

@st.cache_resource
def load_prediction_model():

    model = load_model(
        "models/mnist_cnn.keras"
    )

    return model


model = load_prediction_model()


# =====================================
# DRAWING CANVAS
# =====================================

from streamlit_drawable_canvas import st_canvas


st.subheader("Draw a digit")

canvas_result = st_canvas(
    fill_color="black",
    stroke_width=15,
    stroke_color="white",
    background_color="black",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)


# =====================================
# PREDICT BUTTON
# =====================================

if st.button("🔍 Predict"):

    if canvas_result.image_data is not None:

        # =====================================
        # GET CANVAS IMAGE
        # =====================================

        img = canvas_result.image_data

        # Convert RGBA → grayscale
        img = Image.fromarray(
            img.astype("uint8")
        ).convert("L")


        # =====================================
        # RESIZE TO MNIST SIZE
        # =====================================

        img = img.resize(
            (28, 28)
        )


        # =====================================
        # CONVERT TO NUMPY
        # =====================================

        img_array = np.array(img)


        # =====================================
        # NORMALIZE
        # =====================================

        img_array = (
            img_array.astype("float32") / 255.0
        )


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


        # =====================================
        # MODEL PREDICTION
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


        confidence = (
            np.max(predictions[0]) * 100
        )


        # =====================================
        # DISPLAY RESULT
        # =====================================

        st.success(
            f"Predicted Digit: {predicted_digit}"
        )

        st.info(
            f"Confidence: {confidence:.2f}%"
        )


        # =====================================
        # TOP 3 PREDICTIONS
        # =====================================

        st.subheader(
            "Top 3 Predictions"
        )

        top3_indices = np.argsort(
            predictions[0]
        )[::-1][:3]


        for idx in top3_indices:

            probability = (
                predictions[0][idx] * 100
            )

            st.write(
                f"Digit {idx}: "
                f"{probability:.2f}%"
            )


        # =====================================
        # PROBABILITY CHART
        # =====================================

        st.subheader(
            "Class Probabilities"
        )

        probability_data = {
            str(i): float(predictions[0][i])
            for i in range(10)
        }

        st.bar_chart(
            probability_data
        )