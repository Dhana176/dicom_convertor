# app.py
import streamlit as st
from dicom_converter import convert_single_dicom_to_jpg
# Import your conversion function

st.title("DICOM to JPG Converter")

uploaded_file = st.file_uploader("Upload a DICOM file", type=["dcm"])

if uploaded_file is not None:
    converted_image = convert_single_dicom_to_jpg(uploaded_file, cnn_model)  # Pass your CNN model if needed
    st.image(converted_image, caption="Converted Image", use_column_width=True)
