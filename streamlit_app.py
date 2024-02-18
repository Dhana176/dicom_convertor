# Import necessary libraries
import os
import streamlit as st
import numpy as np
import pydicom
from PIL import Image
from scipy.ndimage import gaussian_filter
import tensorflow as tf

# Function to preprocess a single DICOM file
def preprocess_single_dicom(filepath):
    dicom_data = pydicom.dcmread(filepath)
    image_array = dicom_data.pixel_array.astype(np.float32)
    smoothed_image = gaussian_filter(image_array, sigma=1)
    normalized_image = (smoothed_image - np.min(smoothed_image)) / (np.max(smoothed_image) - np.min(smoothed_image))
    return normalized_image

# Function to create and train CNN
def create_and_train_cnn(input_shape, num_conv_layers=2, filter_size=(3, 3), dropout_rate=0.3, epochs=10):
    model = tf.keras.Sequential()
    for _ in range(num_conv_layers):
        model.add(tf.keras.layers.Conv2D(64, filter_size, activation='relu', padding='same', input_shape=input_shape))
        model.add(tf.keras.layers.MaxPooling2D(2, 2))
        model.add(tf.keras.layers.Dropout(dropout_rate))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Function to convert a single DICOM file to JPG using a trained CNN
def convert_single_dicom_to_jpg(filepath, model):
    dicom_data = pydicom.dcmread(filepath)
    image_array = dicom_data.pixel_array.astype(np.float32)
    normalized_image = (image_array - np.min(image_array)) / (np.max(image_array) - np.min(image_array))
    input_image = normalized_image.reshape(1, *normalized_image.shape, 1)
    
    prediction = model.predict(input_image)
    
    if prediction[0, 0] > 0.5:
        img = Image.fromarray((normalized_image * 255).astype(np.uint8))
        return img

# Streamlit app code
def main():
    st.title("DICOM to JPG Converter")

    # File upload widget
    uploaded_file = st.file_uploader("Upload a DICOM file", type=["dcm"])

    if uploaded_file is not None:
        # Preprocess a single DICOM file
        single_dicom_image = preprocess_single_dicom(uploaded_file)
        
        # Create and train CNN
        input_shape = single_dicom_image.shape + (1,)
        cnn_model = create_and_train_cnn(input_shape, num_conv_layers=2, filter_size=(3, 3), dropout_rate=0.3, epochs=10)

        # Convert a single DICOM to JPG using the trained CNN
        converted_image = convert_single_dicom_to_jpg(uploaded_file, cnn_model)

        # Display the converted image
        st.image(converted_image, caption="Converted Image", use_column_width=True)

if __name__ == "__main__":
    main()
