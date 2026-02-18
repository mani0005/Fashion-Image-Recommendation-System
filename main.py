import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from numpy.linalg import norm

IMAGE_DIR = "images"

# Load ResNet50 model
base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(224,224,3))
base_model.trainable = False

model = tf.keras.Sequential([
    base_model,
    GlobalMaxPooling2D()
])

def extract_features(img_path):
    img = image.load_img(img_path, target_size=(224,224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    features = model.predict(img_array, verbose=0).flatten()
    return features / norm(features)

filenames = []
feature_list = []

print("🔄 Extracting features...")

for file in os.listdir(IMAGE_DIR):
    if file.lower().endswith((".jpg", ".png", ".jpeg")):
        path = os.path.join(IMAGE_DIR, file)
        filenames.append(path)
        feature_list.append(extract_features(path))

# Save features
pickle.dump(feature_list, open("embeddings.pkl", "wb"))
pickle.dump(filenames, open("filenames.pkl", "wb"))

print("✅ Feature extraction completed")
print("Total images processed:", len(filenames))
