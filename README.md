# Fashion Image Recommendation System

A **content-based fashion image recommendation system** that suggests visually similar fashion products based on an uploaded image.  
The system uses **deep learning (ResNet50)** for feature extraction and **K-Nearest Neighbors (KNN)** for similarity matching, with an interactive **Streamlit web interface**.

---

## Dataset

https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small

---

## Features

- Upload an image and get visually similar fashion items
- Deep feature extraction using **ResNet50 (Transfer Learning)**
- Fast similarity search using **KNN (Euclidean Distance)**
- Precomputed embeddings for efficient inference
- Simple and clean **Streamlit UI**

---

## How It Works

1. Fashion images are preprocessed and resized to `224 × 224`
2. A pre-trained **ResNet50** model extracts deep visual features
3. Feature vectors are normalized and stored using Pickle
4. **KNN** finds the top 5 visually similar images
5. Results are displayed in a Streamlit web app

---

## Tech Stack

**Programming Language**
- Python 3.x

**Libraries & Frameworks**
- TensorFlow & Keras
- ResNet50 (Keras Applications)
- Scikit-learn
- NumPy
- PIL (Python Imaging Library)
- Pickle
- Streamlit

