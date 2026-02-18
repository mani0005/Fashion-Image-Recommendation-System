# app.py
import os

import streamlit as st
import pickle
import numpy as np
from PIL import Image
from sklearn.neighbors import NearestNeighbors
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.layers import GlobalMaxPooling2D
import tensorflow as tf
from numpy.linalg import norm

# Page config
st.set_page_config(
    page_title="Fashion Recommendation",
    layout="wide"
)

# -------------------------------
# Load ResNet50 model
# -------------------------------
base_model = ResNet50(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)
base_model.trainable = False

model = tf.keras.Sequential([
    base_model,
    GlobalMaxPooling2D()
])

# -------------------------------
# Load saved embeddings
# -------------------------------
features = pickle.load(open("embeddings.pkl", "rb"))
filenames = pickle.load(open("filenames.pkl", "rb"))

neighbors = NearestNeighbors(n_neighbors=6, metric="euclidean")
neighbors.fit(features)

# -------------------------------
# Feature extraction function
# -------------------------------
def extract_features_from_image(img: Image.Image):
    img = img.resize((224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    feature = model.predict(img_array, verbose=0).flatten()
    return feature / norm(feature)

# -------------------------------
# UI
# -------------------------------
st.title("Fashion Image Recommendation System")
st.write("Upload a fashion image to find visually similar items.")

uploaded_file = st.file_uploader(
    "Upload a fashion image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    col1, col2 = st.columns([1, 3])

    # Uploaded image
    with col1:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Uploaded Image", width=300)

    # Find similar images
    query_features = extract_features_from_image(img)
    distances, indices = neighbors.kneighbors([query_features])

    with col2:
        st.subheader("🔍 Similar Fashion Items")
        cols = st.columns(5)

        # Skip first image (it’s the same image)
        for i, idx in enumerate(indices[0][1:]):
            with cols[i]:
                st.image(filenames[idx], width=180)
