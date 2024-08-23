# MNIST Digit Recognition Project

## Overview
This repository contains a complete MNIST digit recognition project that includes a Streamlit dashboard, a neural network model, and a Jupyter notebook. The project demonstrates the end-to-end process of training a neural network on the MNIST dataset and deploying it through a user-friendly interface.

## Project Structure
- **app.py:** Streamlit application for interactive digit recognition.
- **mnist_model.h5:** Trained neural network model saved in HDF5 format.
- **mnist_digit_recognition_notebook.ipynb:** Jupyter notebook for data exploration, model training, and evaluation.
- **requirements.txt:** List of Python packages required to run the project.
- **data/**: Directory for storing any dataset files (if needed).
- **images/**: Directory for storing images like profile pictures.

## Model Information
The MNIST Digit Recognition model is a feedforward neural network trained on the MNIST dataset, which consists of handwritten digits from 0 to 9. Key model details:
- **Model Type:** Feedforward Neural Network
- **Architecture:** 2 Hidden Layers
- **Activation Functions:** ReLU (Hidden Layers), Softmax (Output Layer)
- **Training Epochs:** 15
- **Batch Size:** 200

## How to Run the Dashboard
1. **Clone the Repository:**
    ```bash
    git clone https://github.com/yourusername/mnist-digit-recognition-project.git
    ```
2. **Navigate to the Project Directory:**
    ```bash
    cd mnist-digit-recognition-project
    ```
3. **Install Dependencies:**
    Create a `requirements.txt` file with the following content:
    ```
    streamlit
    tensorflow
    pillow
    numpy
    matplotlib
    jupyter
    ```
    Install the dependencies using pip:
    ```bash
    pip install -r requirements.txt
    ```
4. **Run the Streamlit App:**
    ```bash
    streamlit run app.py
    ```

## How to Use the Jupyter Notebook
1. **Install Jupyter Notebook (if not installed):**
    ```bash
    pip install jupyter
    ```
2. **Open the Notebook:**
    ```bash
    jupyter notebook mnist_digit_recognition_notebook.ipynb
    ```
3. **Run the Cells:**
    Follow the instructions in the notebook to explore the data, train the model, and evaluate performance.

## About Me
**Ahmad Ali Rafique**  
AI & Machine Learning Specialist

I am an AI and Machine Learning specialist dedicated to developing innovative solutions using advanced machine learning techniques. My expertise includes building and deploying models for various applications, with a focus on creating impactful and user-friendly solutions.

### Contact Information
- [LinkedIn](https://www.linkedin.com/in/ahmad-ali-rafique/)
- [GitHub](https://github.com/Ahmad-Ali-Rafique)
- [Email](arsbussiness@gmail.com)

Feel free to connect with me or reach out if you have any questions or opportunities for collaboration!

