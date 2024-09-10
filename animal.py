import os
import streamlit as st
import keras
from keras.models import load_model
import tensorflow as tf
import numpy as np

st.header('ANIMAL CLASSIFICATION')
st.write("""Upload Animal pic of only CAT,DOG,ELEPHANT,HORSE,LION,PANDA,TIGER""")

animal_names=['cat', 'dog', 'elephant', 'horse', 'lion', 'panda', 'tiger']

model = load_model('animal.h5')

def  predict_image(image_path):
    test_image = keras.utils.load_img(image_path, target_size = (128, 128))
    test_image = keras.utils.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    result = model.predict(test_image)

    ans=tf.nn.softmax(result[0])
    animal=animal_names[np.argmax(ans)]
    return animal

uploaded_file = st.file_uploader('UPLOAD IMAGE')

if uploaded_file is not None:
    with open(os.path.join('upload', uploaded_file.name), 'wb') as f:
        f.write(uploaded_file.getbuffer())
        
    st.image(uploaded_file, width=200)
    st.markdown(predict_image(os.path.join('upload', uploaded_file.name)))
