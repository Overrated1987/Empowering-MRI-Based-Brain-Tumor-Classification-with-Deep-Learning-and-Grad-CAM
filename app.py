import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from io import BytesIO
from PIL import Image

# Model Prediction
labels = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']


def model_prediction(image_data):
    model = load_model('./bestresnet4.h5')
    img = Image.open(BytesIO(image_data))
    img = img.resize((200, 200))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = img_array / 255.0  # Scaling is important
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)
    predicted_class = labels[np.argmax(predictions[0])]
    confidence = round(100 * np.max(predictions[0]), 2)
    return predicted_class, confidence


# Sidebar
st.sidebar.title('Detection System Dashboard')
app_mode = st.sidebar.selectbox('Select Page', ['Home', 'About Dataset', 'Disease Recognition'])

# Home Page
if app_mode == 'Home':
    st.header('Brain Tumor Detection System')
    st.markdown("""
    Welcome to the Brain Tumor Detection System! 🌿🔍
    
    Brain tumors are abnormal growths of cells within the brain, categorized into two types: malignant and benign. Malignant tumors can be life-threatening due to their location and growth rate, making timely and accurate detection crucial. This project aims to distinguish between three types of brain tumors and normal cases (i.e., no tumor) based on their location.
    
    ### How It Works
    1. **Upload Image:** Navigate to the **Disease Recognition** page and upload an image of the suspected tumor.
    2. **Analysis:** The system processes the image using advanced algorithms to classify the tumor.
    3. **Results:** View the classification results and recommendations for further action.
    
    ### Get Started
    Click on the **Disease Recognition** page in the sidebar to upload an image and experience the capabilities of the Brain Tumor Detection System! 
    """)


# About Project
elif app_mode == "About Dataset":
    st.header("About the Brain Tumor MRI Dataset")
    st.markdown("""
    The dataset used for this model is sourced from the Brain Tumor MRI Dataset available on Kaggle, which comprises 7,023 Magnetic Resonance Imaging (MRI) scans. 
    These images are categorized into one of four classifications: Glioma tumors, Meningioma tumors, Pituitary tumors, and images with no tumor.

    | Dataset          | Training | Validation | Testing |
    |------------------|----------|------------|---------|
    | Glioma tumor     | 1060     | 261        | 300     |
    | Meningioma tumor | 1072     | 267        | 306     |
    | Pituitary tumor  | 1158     | 299        | 300     |
    | No tumor         | 1279     | 316        | 405     |
    | **Total**        | **4569** | **1143**   | **1311**|

    **Table 1: Dataset Partition for Different Types of Brain Tumors**

    The distribution of images in the testing dataset is as follows:
    - Pituitary tumor: 300 images
    - Meningioma tumor: 306 images
    - Glioma tumor: 300 images
    - No tumor: 405 images
    """)
    # Displaying the image
    st.image("./BrainTumorType.png",
             caption='Figure 1: The Brain Tumor MRI Types', use_column_width=True)

elif app_mode == "Disease Recognition":
    st.header("Brain Tumor Recognition")
    test_image = st.file_uploader("Choose an Image:", type=["jpg", "jpeg", "png"])
    if test_image is not None:
        st.image(test_image, caption='Brain Tumor MRI Image Uploaded ', width=300)
        if st.button("Predict"):
            with st.spinner("Please wait..."):
                predicted_class, confidence = model_prediction(test_image.read())
                st.success(f"The model is predicting it's a {predicted_class}")
                st.success(f"Confidence in prediction is {confidence}%")
                st.image("./ExplainPrediction.png",
                         caption='Visualization of Model Decision Making', width=300)


# Uncomment the following line if running in a script
# streamlit run E:\source_code\Multiclass_Brain_tumour_Classification_using_Resnet-50-main\app.py
