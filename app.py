import streamlit as st
import pandas as pd
import numpy as np
import joblib

def predict(data):
    clf = joblib.load("rf_model.sav", mmap_mode='r') 
    return clf.predict(data)

# Function to map classes to images
def class_to_image(class_name):
    if class_name == "setosa":
        return "images/setosa.jpg"  
    elif class_name == "versicolor":
        return "images/versicolor.jpg"
    elif class_name == "virginica":
        return "images/virginica.jpg"

st.title('Classifying Iris Flowers')
st.markdown('Model to classify iris flowers into '
            '(setosa, versicolor, virginica) based on their sepal/petal '
            'length/width.')

st.header("Plant Features")
col1, col2 = st.columns(2)

with col1:
    st.text("Sepal characteristics")
    sepal_l = st.slider('Sepal length (cm)', 1.0, 8.0, 0.5)
    sepal_w = st.slider('Sepal width (cm)', 2.0, 4.4, 0.5)

with col2:
    st.text("Petal characteristics")
    petal_l = st.slider('Petal length (cm)', 1.0, 7.0, 0.5)
    petal_w = st.slider('Petal width (cm)', 0.1, 2.5, 0.5)

st.text('')
if st.button("Predict type of Iris"):
    # Predict returns an array of numeric classes (e.g., [0], [1], or [2])
    result = predict(np.array([[sepal_l, sepal_w, petal_l, petal_w]]))

    # Map numeric class to the string label
    label_map = {0: "setosa", 1: "versicolor", 2: "virginica"}
    predicted_label = label_map[result[0]]
    
    st.text(predicted_label)

    # Display the image/icon corresponding to the predicted class
    image_path = class_to_image(predicted_label)
    st.image(image_path, use_container_width=True)

st.text('')
