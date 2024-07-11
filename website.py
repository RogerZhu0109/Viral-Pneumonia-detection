import pandas as pd
import streamlit as st
from keras.models import load_model
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

model = load_model('model.h5')

labels = ['Lung_Opacity', 'Normal', 'Viral Pneumonia']

st.title('Viral Pneumonia Detection')
st.write('upload image for classification')

file = st.file_uploader("", type=["jpg", "png"])

if file:
    img = tf.keras.preprocessing.image.load_img(file, target_size=(100, 100), color_mode='grayscale')
    x = tf.keras.preprocessing.image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    prediction = model.predict(x)

    bar_data = prediction.flatten()
    data = pd.DataFrame({
        'index': labels,
        'categories': bar_data,
    }).set_index('index')
    st.bar_chart(data)


    st.sidebar.title(labels[np.argmax(prediction)])
    st.sidebar.image(img, width=300)
    st.sidebar.success("Result: " + labels[np.argmax(prediction)])
    st.sidebar.info("confidence: " + str(max(prediction[0])*100) + "%")

