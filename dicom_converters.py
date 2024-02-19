# coding: utf-8
import os
import numpy as np
import tensorflow as tf
import pydicom
from PIL import Image
from scipy.ndimage import gaussian_filter
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Import requests and scipy.optimize if needed
import requests
def create_and_train_cnn(input_shape, num_conv_layers=2, filter_size=(3, 3), dropout_rate=0.3, epochs=10):
    model = tf.keras.Sequential()

    # Add convolutional layers
    for _ in range(num_conv_layers):
        model.add(tf.keras.layers.Conv2D(64, filter_size, activation='relu', padding='same', input_shape=input_shape))
        model.add(tf.keras.layers.MaxPooling2D(2, 2))
        model.add(tf.keras.layers.Dropout(dropout_rate))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # Add data augmentation
    data_generator = ImageDataGenerator(rotation_range=15, width_shift_range=0.1, height_shift_range=0.1, shear_range=0.1, zoom_range=0.1, horizontal_flip=True)
    
    # Example: Assuming you have train_data and validation_data
    # model.fit(data_generator.flow(train_data, train_labels), epochs=epochs, validation_data=(validation_data, validation_labels))
    
    return model
def preprocess_single_dicom(filepath):
    dicom_data = pydicom.dcmread(filepath)
    image_array = dicom_data.pixel_array.astype(np.float32)

    # Apply Gaussian filter to remove noise
    smoothed_image = gaussian_filter(image_array, sigma=1)

    normalized_image = (smoothed_image - np.min(smoothed_image)) / (np.max(smoothed_image) - np.min(smoothed_image))
    return normalized_image
def convert_single_dicom_to_jpg(filepath, output_directory, model):
    dicom_data = pydicom.dcmread(filepath)
    image_array = dicom_data.pixel_array.astype(np.float32)
    normalized_image = (image_array - np.min(image_array)) / (np.max(image_array) - np.min(image_array))
    input_image = normalized_image.reshape(1, *normalized_image.shape, 1)
    
    prediction = model.predict(input_image)
    
    if prediction[0, 0] > 0.5:
        save_path = os.path.join(output_directory, f"{os.path.basename(filepath)[:-4]}.jpg")
        img = Image.fromarray((normalized_image * 255).astype(np.uint8))
        img.save(save_path)

# Example usage for a single DICOM file
dicom_file_path = "C:/Users/ktpdharani/dicomfiles/dicnew1/input/ex.dcm"
jpg_output_directory = "C:/Users/ktpdharani/dicomfiles/dicnew1/output"

# Preprocess a single DICOM file
single_dicom_image = preprocess_single_dicom(dicom_file_path)

# Create and train CNN
input_shape = single_dicom_image.shape + (1,)
cnn_model = create_and_train_cnn(input_shape, num_conv_layers=2, filter_size=(3, 3), dropout_rate=0.3, epochs=10)

# Convert a single DICOM to JPG using the trained CNN and save in the output folder
convert_single_dicom_to_jpg(dicom_file_path, jpg_output_directory, cnn_model)
get_ipython().run_line_magic('load', 'dicom_convertor.ipynb')
# Save the notebook as a Python script
