import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf


CLASSES = ['glioma', 'meningioma', 'notumor', 'pituitary']

model_path = "mri-image-classifier.keras"
default_img_path = 'default_img.png'

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(model_path, compile=False)

target_size = (384, 384)
model = load_model()
st.title("Brain Tumor MRI Classifier🧠")
default_img_col, upload_col = st.columns([0.6,0.4])


def preprocess_image(image_bytes):
    image = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
    image = tf.cast(image, dtype=tf.float32)
    image = tf.image.resize(image, [*target_size])
    return image


with default_img_col:
    img = st.empty()
    img.image(np.array(Image.open(default_img_path).resize(target_size)))

with upload_col:
    uploaded_file = st.file_uploader("Upload MRI Scan", type=['jpg','jpeg','png'])

if uploaded_file is not None:
    bytes_array = uploaded_file.getvalue()
    image = preprocess_image(bytes_array)
    np_array = np.array(tf.cast(image, tf.uint8))
    img.image(np_array)

    with upload_col:
        pred = st.empty()
        if pred.button("Predict!"):
            model_output = st.empty()
            with st.spinner("model analyzing...", show_time=True):
                preds = model.predict(tf.expand_dims(image, axis=0))[0]
                prediction, acc = np.argmax(preds), f"{np.max(preds):.4f}"
            
            model_output.write(f"Prediction: {CLASSES[prediction].upper()} | Accuracy: {float(acc)*100}%")