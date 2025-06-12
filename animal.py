from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image
import streamlit as st
import pandas as pd
import time

# Load model
model = load_model("animal.h5", compile=False)

# Class labels
animal_names = ['cat', 'dog', 'elephant', 'horse', 'lion', 'panda', 'tiger']

# Preprocess uploaded image
def preprocess(img):
    img = img.resize((64, 64))
    img_array = image.img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

# Page config
st.set_page_config(page_title="Animal Image Classifier", layout="wide")

# Custom CSS
st.markdown("""
    <style>
        body {
            background-color: #000000;
        }
        .stApp {
            background-color: #000000;
            color: white;
            padding: 0;
        }
        h1, h3, p {
            text-align: center;
            color: white;
        }
        .title {
            font-size: 3em;
            font-weight: bold;
            margin-top: 0.5em;
            margin-bottom: 0.2em;
            color: #00FFAA;
        }
        .subtitle {
            font-size: 1.2em;
            color: #AAAAAA;
        }
        .uploaded-img {
            display: flex;
            justify-content: center;
        }
        .stImage > img {
            border-radius: 15px;
            box-shadow: 0 0 20px #00FFAA;
        }
        .prediction {
            font-size: 1.8em;
            font-weight: bold;
            color: #00FFAA;
            text-align: center;
            animation: fadeInScale 1.2s ease-out;
        }
        .confidence {
            text-align: center;
            font-size: 1.2em;
            color: #00FFDD;
            animation: fadeIn 1s ease-in;
        }
        .stProgress > div > div > div {
            background-color: #00ff99 !important;
        }
        .white-table td, .white-table th {
            color: white !important;
        }

        @keyframes fadeInScale {
            0% { opacity: 0; transform: scale(0.9); }
            100% { opacity: 1; transform: scale(1); }
        }
        @keyframes fadeIn {
            0% { opacity: 0; }
            100% { opacity: 1; }
        }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1 class='title'>🐾 Animal Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Upload an animal image and let the AI guess with confidence!</p>", unsafe_allow_html=True)

# Upload file
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)

    st.markdown("<div class='uploaded-img'>", unsafe_allow_html=True)
    st.image(img, caption="🖼️ Uploaded Image", width=300)
    st.markdown("</div>", unsafe_allow_html=True)

    with st.spinner("🔍 Classifying..."):
        time.sleep(1)
        result = model.predict(preprocess(img))
        index = np.argmax(result)
        confidence = float(np.max(result)) * 100
        predicted_class = animal_names[index]

    # Animated prediction output
    st.markdown(f"<div class='prediction'>Prediction: {predicted_class.upper()}</div>", unsafe_allow_html=True)
    st.progress(int(confidence))
    st.markdown(f"<div class='confidence'>Confidence: <strong>{confidence:.2f}%</strong></div>", unsafe_allow_html=True)

    # Optional: show all probabilities
    with st.expander("📊 Show all class probabilities"):
        probs = result[0]
        prob_data = {animal_names[i]: probs[i] * 100 for i in range(len(animal_names))}

        # Display white table
        st.markdown('<div class="white-table">', unsafe_allow_html=True)
        st.table({k: f"{v:.2f}%" for k, v in prob_data.items()})
        st.markdown('</div>', unsafe_allow_html=True)

        # Optional animated bar chart
        st.subheader("📈 Probability Bar Chart")
        df = pd.DataFrame({
            'Animal': animal_names,
            'Confidence': [round(p * 100, 2) for p in probs]
        })
        st.bar_chart(df.set_index('Animal'))
