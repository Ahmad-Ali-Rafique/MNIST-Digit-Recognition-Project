import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image, ImageOps

# Load the trained model
model = load_model('mnist_model.h5')

# Define a function to preprocess the uploaded image
def preprocess_image(image):
    image = ImageOps.grayscale(image)
    image = image.resize((28, 28))
    image = np.array(image)
    image = image.reshape(1, 28 * 28).astype('float32')
    image /= 255.0
    return image

# Sidebar for navigation
st.sidebar.title("Welcome to My Dashboard")

# Centered profile picture and name
st.sidebar.image('Ahmad Ali Profile Photo.png', use_column_width=True, caption="Ahmad Ali Rafique")
st.sidebar.write("**AI & Machine Learning Specialist**")

# About the Model section
st.sidebar.header("About the Model")
st.sidebar.write("""
    The MNIST Digit Recognition model is a sophisticated neural network designed to classify handwritten digits from 0 to 9. It is built on the MNIST dataset, which comprises thousands of digit images.
    
    **Model Details:**
    - **Type:** Feedforward Neural Network
    - **Architecture:** 2 Hidden Layers
    - **Activation Functions:** ReLU (Hidden Layers), Softmax (Output Layer)
    - **Training Epochs:** 15
    - **Batch Size:** 200
""")

# Contact information
st.sidebar.header("Contact Information")
st.sidebar.write("Feel free to reach out through the following channels:")
st.sidebar.write("[LinkedIn](https://www.linkedin.com/in/ahmad-ali-rafique/)")
st.sidebar.write("[GitHub](https://github.com/Ahmad-Ali-Rafique/)")
st.sidebar.write("[Email](arsbussiness786@gmail.com)")

# Main section for the app
st.title("MNIST Digit Recognition")

st.write("Upload a digit image to classify it.")

# File uploader for image input
uploaded_file = st.file_uploader("Choose an image...", type="png")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess the image for the model
    processed_image = preprocess_image(image)
    
    # Predict the digit
    prediction = model.predict(processed_image)
    predicted_digit = np.argmax(prediction)
    
    # Display the predicted digit
    st.write(f"**Predicted Digit:** {predicted_digit}")

# Button to make a prediction
if st.button('Predict'):
    if uploaded_file is not None:
        st.write("Prediction Complete")
    else:
        st.write("Please upload an image first.")
